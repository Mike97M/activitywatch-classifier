"""Benchmark the classification pipeline the way production runs it.

Every case goes through the production path: anonymizer -> deterministic rules -> LLM oracle.
Reported separately:
  * LLM accuracy on all cases, and on the cases the rules don't cover (the LLM's real workload)
  * rule accuracy on the cases the rules cover
  * end-to-end accuracy (rule if it fires, else LLM)
  * seconds per item and a confidence-threshold table (would a label be learned, and was it right?)

    uv run python bench/run_benchmark.py                    # production JSON mode
    uv run python bench/run_benchmark.py --mode options     # single-pass option-probability readout
    uv run python bench/run_benchmark.py --corrections      # also use your human corrections (stay local)

Synthetic cases live in bench/cases.jsonl. Results go to data/benchmark_v2_results.json (git-ignored).
Activity titles are printed for synthetic cases only, never for your own corrections.
"""

import argparse
import concurrent.futures
import json
import logging
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anonymizer import anonymizer  # noqa: E402
from classifier import FastClassifier  # noqa: E402
from config import DATA_DIR, DEFAULT_TAXONOMY, TRAINING_DATA_PATH  # noqa: E402
from gemma_oracle import GemmaOracle  # noqa: E402
from llama_server_manager import llama_server  # noqa: E402

CASES_PATH = Path(__file__).resolve().parent / "cases.jsonl"
RESULTS_PATH = DATA_DIR / "benchmark_v2_results.json"
LETTERS = "ABCDEFGHIJKLMNOP"
THRESHOLDS = (0.5, 0.7, 0.9, 0.95)


def load_cases(with_corrections: bool, path: Path = CASES_PATH):
    source = "synthetic" if path == CASES_PATH else "real"  # titles of real cases are never printed
    cases = [dict(json.loads(line), source=source) for line in path.open(encoding="utf-8") if line.strip()]
    if with_corrections and TRAINING_DATA_PATH.exists():
        field = re.compile(r"(?:^|\s)(app|title|url):(.*?)(?=\s(?:package|domain|root_domain|path|title|url):|$)", re.S)
        for line in TRAINING_DATA_PATH.open(encoding="utf-8"):
            row = json.loads(line) if line.strip() else {}
            if row.get("source") != "human_correction":
                continue
            f = {m.group(1): m.group(2).strip() for m in field.finditer(row["text"])}
            if f.get("app") or f.get("title"):
                cases.append({"app": f.get("app", ""), "title": f.get("title", ""), "url": f.get("url", ""),
                              "expected": row["full_category"], "source": "human"})
    return cases


def production_input(case):
    """What the pipeline hands to the rules and the LLM (pipeline.py: sanitize, title falls back to app)."""
    clean = anonymizer.sanitize_event_data({"app": case["app"], "title": case["title"], "url": case["url"]})
    if not clean.get("title") and clean.get("app"):
        clean["title"] = clean["app"]
    return clean


def option_readout(oracle, url, clean, short=False):
    """Two lettered passes (category, then subcategory); confidence = P(cat) * P(sub | cat).

    short=True lists bare category names: the taxonomy is already in the (cached) prompt prefix,
    so the uncached part after the activity shrinks from ~250 to ~110 tokens.
    """
    p = oracle._build_prompt(clean.get("app", ""), clean.get("title", ""), clean.get("url"), DEFAULT_TAXONOMY)
    head = p.split("Respond with ONLY")[0].rstrip()

    def read(question, options):
        opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))
        prompt = f"{head}\n\n{question}\nOptions:\n{opts}\n\nRespond with only the letter of the best option."
        r = httpx.post(f"{url}/chat/completions", timeout=900, json={
            "messages": [{"role": "user", "content": prompt}], "max_tokens": 1, "temperature": 0,
            "logprobs": True, "top_logprobs": 40})
        r.raise_for_status()
        seen = {}
        for e in r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
            tok = e["token"].strip()
            if len(tok) == 1 and tok in LETTERS:
                seen[tok] = max(seen.get(tok, -1e9), e["logprob"])
        floor = min(seen.values(), default=0.0) - 2.0
        lps = [seen.get(LETTERS[i], floor) for i in range(len(options))]
        w = [math.exp(z - max(lps)) for z in lps]
        return [x / sum(w) for x in w]

    cats = list(DEFAULT_TAXONOMY)
    cat_options = cats if short else [f"{k} ({', '.join(v)})" for k, v in DEFAULT_TAXONOMY.items()]
    pc = read("Which category best describes this activity?", cat_options)
    ci = max(range(len(cats)), key=pc.__getitem__)
    subs = DEFAULT_TAXONOMY[cats[ci]]
    if len(subs) == 1:
        return f"{cats[ci]} > {subs[0]}", pc[ci]
    ps = read(f"The activity is in the category '{cats[ci]}'. Which subcategory fits best?", subs)
    si = max(range(len(subs)), key=ps.__getitem__)
    return f"{cats[ci]} > {subs[si]}", pc[ci] * ps[si]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("json", "options", "options-short"), default="json")
    ap.add_argument("--corrections", action="store_true", help="add your human corrections as cases")
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--cases", type=Path, default=CASES_PATH,
                    help="JSONL of {app,title,url,expected}; anything but bench/cases.jsonl is treated as private")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cases = load_cases(args.corrections, args.cases)
    oracle = GemmaOracle(backend="llama_server")
    inputs = [production_input(c) for c in cases]
    rules = []
    for clean in inputs:
        r = FastClassifier.match_rules(FastClassifier.extract_features(clean))
        rules.append(" > ".join(r) if r else None)

    try:
        t0 = time.time()
        if args.mode == "json":
            acts = [{"app": c.get("app", ""), "title": c.get("title", ""), "url": c.get("url", "")} for c in inputs]
            res = oracle.classify_batch(acts, max_streams=args.streams)
            llm = [(r["full_category"], float(r.get("confidence", 0.0)), r.get("backend")) for r in res]
        else:
            url = llama_server.ensure_server(oracle.find_local_model_path())
            with concurrent.futures.ThreadPoolExecutor(args.streams) as ex:
                short = args.mode == "options-short"
                llm = [(*x, args.mode) for x in ex.map(lambda c: option_readout(oracle, url, c, short), inputs)]
        seconds = time.time() - t0
    finally:
        llama_server.stop()

    n = len(cases)
    exp = [c["expected"] for c in cases]
    cat = lambda s: s.split(" > ")[0]  # noqa: E731
    miss_idx = [i for i in range(n) if rules[i] is None]
    hit_idx = [i for i in range(n) if rules[i] is not None]

    def acc(idx, pred):
        if not idx:
            return "n/a"
        c = sum(cat(pred(i)) == cat(exp[i]) for i in idx)
        s = sum(pred(i) == exp[i] for i in idx)
        return f"category {100 * c / len(idx):5.1f}%  subcategory {100 * s / len(idx):5.1f}%  (n={len(idx)})"

    llm_pred = lambda i: llm[i][0]  # noqa: E731
    fallbacks = sum(b == "gemma_heuristic_proxy" for _, _, b in llm)
    sources = defaultdict(int)
    for c in cases:
        sources[c["source"]] += 1

    print(f"\nmode={args.mode}  cases={n} ({dict(sources)})  {seconds / n:.2f} s/item  heuristic fallbacks={fallbacks}")
    print(f"LLM, all cases:            {acc(range(n), llm_pred)}")
    print(f"LLM, rule-miss cases:      {acc(miss_idx, llm_pred)}   <- the LLM's real workload")
    print(f"Rules, where they fire:    {acc(hit_idx, lambda i: rules[i])}")
    print(f"End to end (rules -> LLM): {acc(range(n), lambda i: rules[i] or llm[i][0])}")

    print("\nConfidence thresholds on rule-miss cases (labels the pipeline would learn):")
    for th in THRESHOLDS:
        right = sum(llm[i][1] >= th and llm[i][0] == exp[i] for i in miss_idx)
        wrong = sum(llm[i][1] >= th and llm[i][0] != exp[i] for i in miss_idx)
        print(f"  >= {th:<4}: learn {right:3d} right, {wrong:3d} wrong")

    print("\nLLM subcategory accuracy by expected category:")
    by_cat = defaultdict(list)
    for i in range(n):
        by_cat[cat(exp[i])].append(llm[i][0] == exp[i])
    for k, v in sorted(by_cat.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  {k:<26} {sum(v):3d}/{len(v):<3d}")

    print("\nLLM misses (synthetic cases only; real cases and your corrections are counted above but never printed):")
    for i in range(n):
        if llm[i][0] != exp[i] and cases[i]["source"] == "synthetic":
            print(f"  #{i:<3d} {cases[i]['app'][:14]:<14} {cases[i]['title'][:46]:<46} want {exp[i]}  got {llm[i][0]} ({llm[i][1]:.2f})")

    RESULTS_PATH.write_text(json.dumps({
        "mode": args.mode, "seconds_per_item": seconds / n, "n": n,
        "items": [{"index": i, "source": cases[i]["source"], "expected": exp[i], "rule": rules[i],
                   "llm": llm[i][0], "confidence": llm[i][1]} for i in range(n)],
    }, indent=2), encoding="utf-8")
    print(f"\nPer-item results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
