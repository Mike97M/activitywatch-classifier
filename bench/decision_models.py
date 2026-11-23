"""Benchmark small decision models (JevK5-Lite, Laya) on the same cases and inputs as the LLM benchmark.

Needs torch + transformers (+ laya for the Laya model), which the app itself does not depend on,
so run it from a separate environment:

    uv venv .venv-models
    uv pip install --python .venv-models/bin/python --extra-index-url https://download.pytorch.org/whl/cpu \\
        torch transformers sentencepiece safetensors huggingface-hub scikit-learn joblib numpy laya
    .venv-models/bin/python bench/decision_models.py jevk5lite --cases data/real_benchmark.jsonl
    .venv-models/bin/python bench/decision_models.py laya      --cases data/real_benchmark.jsonl
    # as a checker of Gemma's labels (results from bench/run_benchmark.py on the same cases):
    .venv-models/bin/python bench/decision_models.py laya --cases data/real_benchmark.jsonl \
        --gate data/benchmark_results_json.json

Inputs get the production cleaning (anonymizer) and scores are split into all cases and the cases the
rules don't cover, like bench/run_benchmark.py. Only aggregates are printed.

JevK5-Lite inference is re-implemented from jevk5/lite.py (allebee/jevk5 v0.3.1, Apache-2.0), so the
jevk5 package itself is not installed: a DeBERTa-v3 encoder plus a small MLP over [LABEL] marker states.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.disable(logging.CRITICAL)
from anonymizer import anonymizer  # noqa: E402
from app_knowledge import app_knowledge  # noqa: E402
from classifier import FastClassifier  # noqa: E402
from config import DEFAULT_TAXONOMY as TAX  # noqa: E402

CASES_PATH = Path(__file__).resolve().parent / "cases.jsonl"
FULL = [f"{c} > {s}" for c, subs in TAX.items() for s in subs]


def production_input(c):
    d = anonymizer.sanitize_event_data({"app": c["app"], "title": c["title"], "url": c["url"]})
    d["title"] = d.get("title") or d.get("app", "")
    return d


def activity_text(d):
    app, url = d.get("app", ""), d.get("url", "")
    desc = app_knowledge.get_description(app) if app else ""
    domain = urlparse(url if "://" in url else f"https://{url}").netloc.split(":")[0] if url else ""
    return (f"Application: {app or 'browser'}{f' ({desc})' if desc else ''}\n"
            f"Window title: {d.get('title', '')}\nWebsite: {domain or 'none'}\nURL: {url or 'none'}")


class JevK5Lite:
    def __init__(self, threads):
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from torch import nn
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(threads)
        path = Path(snapshot_download("alibiserikbay/JevK5-Lite"))
        cfg = json.loads((path / "lite_config.json").read_text())
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.enc = AutoModel.from_pretrained(path).eval()
        h = self.enc.config.hidden_size
        self.net = nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.Linear(h, 1)).eval()
        self.net.load_state_dict(load_file(path / "scorer.safetensors"))
        self.max_len = int(cfg.get("max_len", 512))
        self.t_single = float(cfg.get("temperature_single", 1.0))
        self.task_id = self.tok.convert_tokens_to_ids("[TASK]")
        self.label_id = self.tok.convert_tokens_to_ids("[LABEL]")

    def _piece(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def heads(self, text, heads):
        """[(task, [labels])] single-label heads -> a probability list per head, from one encoder pass."""
        torch = self.torch
        schema, pos, spans = [], [], []
        for task, labels in heads:
            schema += [self.task_id] + self._piece(f"{task} (one)")
            a = len(pos)
            for lab in labels:
                pos.append(1 + len(schema))
                schema += [self.label_id] + self._piece(lab)
            spans.append((a, len(pos)))
        room = max(16, self.max_len - len(schema) - 3)
        body = self.tok(text, add_special_tokens=False, truncation=True, max_length=room)["input_ids"]
        ids = [self.tok.cls_token_id] + schema + [self.tok.sep_token_id] + body + [self.tok.sep_token_id]
        s = 2 + len(schema)
        with torch.inference_mode():
            x = torch.tensor([ids])
            hs = self.enc(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state[0]
            t = hs[s: max(s + len(body), s + 1)].mean(0, keepdim=True)
            lab = hs[pos]
            z = self.net(torch.cat([lab, t.expand_as(lab), lab * t.expand_as(lab)], -1)).squeeze(-1).float()
        return [torch.softmax(z[a:b] / self.t_single, -1).tolist() for a, b in spans]

    def variants(self):
        def one_head(d):
            (p,) = self.heads(activity_text(d), [("activity type", [f"{s} ({c})" for c, subs in TAX.items() for s in subs])])
            i = max(range(len(p)), key=p.__getitem__)
            return FULL[i], p[i], dict(zip(FULL, p))

        def two_heads(d):
            pc, ps = self.heads(activity_text(d), [
                ("activity category", list(TAX)),
                ("activity subcategory", [f"{s} ({c})" for c, subs in TAX.items() for s in subs])])
            joint, k = {}, 0
            for ci, (c, subs) in enumerate(TAX.items()):
                for s in subs:
                    joint[f"{c} > {s}"] = pc[ci] * ps[k]
                    k += 1
            z = sum(joint.values())
            probs = {k: v / z for k, v in joint.items()}
            best = max(probs, key=probs.get)
            return best, probs[best], probs

        return [("JevK5-Lite, one head (all subcategories)", one_head),
                ("JevK5-Lite, category x subcategory heads", two_heads)]


class Laya:
    def __init__(self, threads):
        import torch
        from laya import Router

        torch.set_num_threads(threads)
        self.router = Router()  # lazy: only the checkpoint a text is routed to (preload=True also loads a vision model, ~4.6 GB)

    def _ans(self, r, q):
        a = r["answers"][q]
        p = {k: float(v) for k, v in (a.get("probabilities") or {}).items()}
        return a["choice"], p.get(a["choice"], float(a.get("confidence", 0.0))), p

    def variants(self):
        flat_q = {"activity": {"type": "choice", "instructions": "Which kind of computer activity is this?",
                               "criteria": {f"{c} > {s}": f"{s} ({c})" for c, subs in TAX.items() for s in subs}}}
        cat_q = {"category": {"type": "choice", "instructions": "Which category best describes this computer activity?",
                              "criteria": {c: ", ".join(subs) for c, subs in TAX.items()}}}

        def flat(d):
            return self._ans(self.router.predict(activity_text(d), flat_q), "activity")

        def two_step(d):
            t = activity_text(d)
            c, pc, _ = self._ans(self.router.predict(t, cat_q), "category")
            subs = TAX[c]
            if len(subs) == 1:
                return f"{c} > {subs[0]}", pc, None
            sq = {"sub": {"type": "choice", "instructions": f"This is a '{c}' activity. Which kind exactly?",
                          "criteria": {s: s for s in subs}}}
            s, ps, _ = self._ans(self.router.predict(t, sq), "sub")
            return f"{c} > {s}", pc * ps, None  # no full distribution: not usable as a --gate checker

        return [("Laya, one question (all subcategories)", flat), ("Laya, category then subcategory", two_step)]


def gate_report(cases, inputs, preds, gemma_items):
    """Gemma labels on rule-miss cases, learned only if this model also gives Gemma's label P >= t."""
    miss = [i for i in range(len(cases)) if FastClassifier.match_rules(FastClassifier.extract_features(inputs[i])) is None]
    print("  as a checker of Gemma's labels (rule-miss cases, Gemma confidence >= 0.95 in all rows):")
    for t in (None, 0.1, 0.2, 0.3, 0.5):
        ok = lambda i: gemma_items[i]["confidence"] >= 0.95 and (t is None or preds[i][2].get(gemma_items[i]["llm"], 0.0) >= t)  # noqa: E731
        r = sum(ok(i) and gemma_items[i]["llm"] == cases[i]["expected"] for i in miss)
        w = sum(ok(i) and gemma_items[i]["llm"] != cases[i]["expected"] for i in miss)
        name = "Gemma alone" if t is None else f"+ checker P(Gemma's label) >= {t}"
        print(f"    {name:<36} learn {r:3d} right, {w:3d} wrong  ({100 * r / max(1, r + w):5.1f}% correct)")


def report(name, cases, inputs, preds, seconds):
    n = len(cases)
    cat = lambda s: s.split(" > ")[0]  # noqa: E731
    miss = [i for i in range(n) if FastClassifier.match_rules(FastClassifier.extract_features(inputs[i])) is None]

    def acc(idx):
        if not idx:
            return "n/a (no cases)"
        c = sum(cat(preds[i][0]) == cat(cases[i]["expected"]) for i in idx)
        s = sum(preds[i][0] == cases[i]["expected"] for i in idx)
        return f"category {100 * c / len(idx):5.1f}%  subcategory {100 * s / len(idx):5.1f}%  (n={len(idx)})"

    print(f"\n{name}: {seconds / n:.3f} s/item")
    print(f"  all cases:       {acc(range(n))}")
    print(f"  rule-miss cases: {acc(miss)}   <- what the model would actually see")
    for th in (0.3, 0.5, 0.7, 0.9):
        r = sum(preds[i][1] >= th and preds[i][0] == cases[i]["expected"] for i in miss)
        w = sum(preds[i][1] >= th and preds[i][0] != cases[i]["expected"] for i in miss)
        print(f"  rule-miss, confidence >= {th}: {r:3d} right, {w:3d} wrong")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=["jevk5lite", "laya"])
    ap.add_argument("--cases", type=Path, default=CASES_PATH)
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4),
                    help="CPU threads (default 8: on hybrid CPUs the efficiency cores slow torch down)")
    ap.add_argument("--gate", type=Path, help="per-item Gemma results from bench/run_benchmark.py on the same cases")
    a = ap.parse_args()
    cases = [json.loads(line) for line in a.cases.open(encoding="utf-8") if line.strip()]
    gemma = json.loads(a.gate.read_text())["items"] if a.gate else None
    if gemma is not None and len(gemma) != len(cases):
        sys.exit(f"--gate results have {len(gemma)} items but --cases has {len(cases)}; use the same cases file")
    inputs = [production_input(c) for c in cases]
    model = (JevK5Lite if a.model == "jevk5lite" else Laya)(a.threads)
    print(f"{a.model}: {len(cases)} cases, {a.threads} CPU threads")
    for name, fn in model.variants():
        fn(inputs[0])  # warm-up
        t0 = time.time()
        preds = [fn(d) for d in inputs]
        report(name, cases, inputs, preds, time.time() - t0)
        if gemma is not None and preds and preds[0][2] is not None:
            gate_report(cases, inputs, preds, gemma)


if __name__ == "__main__":
    main()
