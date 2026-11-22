"""Fast ML Classifier with Calibrated Confidence Scoring, Hierarchical Taxonomy, and Online Active Learning."""

import functools
import json
import logging
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from anonymizer import anonymizer
from app_knowledge import app_knowledge
from category_sync import (
    GENERIC_DOMAINS,
    GENERIC_HOST_APPS,
    SUBCATEGORY_SEEDS,
    TERMINAL_APPS,
)
from config import (
    DEFAULT_TAXONOMY,
    MODEL_PATH,
    TRAINING_DATA_PATH,
    flatten_taxonomy,
    split_category_path,
)

logger = logging.getLogger("classifier")

# Deterministic rules run before the ML model and the LLM oracle. They use the same
# regex bank that is synced into ActivityWatch, so both views agree on obvious cases.
_COMPILED_SEEDS: Dict[str, Dict[str, re.Pattern]] = {
    cat: {sub: re.compile(pat, re.I) for sub, pat in subs.items()}
    for cat, subs in SUBCATEGORY_SEEDS.items()
}

# High-precision overrides for domains that the general seed bank gets wrong.
_DOMAIN_OVERRIDES = [
    (re.compile(r"(^|\.)docs\.google\.com$", re.I), "Productivity & Planning", "Docs & Spreadsheets"),
    (re.compile(r"(^|\.)music\.youtube\.com$", re.I), "Media & Entertainment", "Music & Audio"),
    (re.compile(r"(^|\.)(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?$", re.I), "Development", "Coding & IDE"),
]

_FEATURE_FIELD_RE = re.compile(
    r"(?:^|\s)(app|package|domain|root_domain|path|title|url):(.*?)(?=\s(?:app|package|domain|root_domain|path|title|url):|$)",
    re.S,
)


def _feature_fields(feature_text: str) -> Dict[str, str]:
    """Split the structured feature string back into app/domain/title/url fields."""
    return {m.group(1): m.group(2).strip() for m in _FEATURE_FIELD_RE.finditer(feature_text)}


def _seed_match(text: str) -> Optional[Tuple[str, str]]:
    """Return the first (category, subcategory) whose seed regex matches, in taxonomy order."""
    if not text:
        return None
    for cat, subs in _COMPILED_SEEDS.items():
        for sub, pat in subs.items():
            if pat.search(text):
                return cat, sub
    return None


def is_email_feature_text(feature_text: str) -> bool:
    """Identify email clients, webmail domains, and inbox titles to prevent misclassification as browsing."""
    f_lower = feature_text.lower()
    
    if re.search(r'\bapp:(thunderbird|evolution|kmail|mutt|neomutt|alpine|k9mail|k-9 mail|geary|claws-mail|mailspring)\b', f_lower):
        return True
    if any(p in f_lower for p in ['package:com.google.android.gm', 'package:com.yahoo.mobile.client.android.mail', 'package:ch.protonmail.android', 'package:de.heinlein.mailbox']):
        return True

    email_domains = (
        'mail.google.com', 'gmail.com', 'mailbox.org', 'app.mailbox.org',
        'account.mailbox.org', 'proton.me', 'mail.proton.me', 'protonmail.com',
        'outlook.live.com', 'outlook.office.com', 'outlook.office365.com',
        'mail.yahoo.com', 'mail.aol.com', 'mail.zoho.com', 'fastmail.com',
        'app.fastmail.com', 'tuta.com', 'tutanota.com', 'mail.tuta.com',
    )
    if any(f'domain:{d}' in f_lower or f'root_domain:{d}' in f_lower or f'url:https://{d}' in f_lower or f'url:http://{d}' in f_lower for d in email_domains):
        return True
    if 'domain:webmail.' in f_lower or 'root_domain:webmail.' in f_lower:
        return True

    if re.search(r'\btitle:[^u]*\b(gmail|proton\s*mail|mailbox\.org|app\.mailbox\.org|yahoo\s*mail|fastmail|k9mail)\b', f_lower):
        return True
    if 'outlook' in f_lower and any(w in f_lower for w in ['mail', 'inbox', 'drafts', 'sent', 'message']):
        return True
    if re.search(r'\btitle:[^u]*\b(inbox|unread|drafts|sent mail|spam|compose)\b', f_lower) and any(
        k in f_lower for k in ['mail', 'outlook', 'google', 'office', 'yahoo', 'proton', 'email']
    ):
        return True

    return False


def _locked(method):
    """Serialize access to training samples, vectorizer and model across threads."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class FastClassifier:
    """Fast, local classifier providing real-time hierarchical predictions and calibrated confidence scores."""

    def __init__(
        self,
        taxonomy: Optional[Dict[str, List[str]]] = None,
        categories: Optional[List[str]] = None,
        model_path: Path = MODEL_PATH,
        training_data_path: Path = TRAINING_DATA_PATH,
    ):
        # Pipeline runs in a worker thread while the web UI submits corrections (which retrain).
        # ponytail: one lock for everything; predictions wait while a retrain runs (seconds).
        self._lock = threading.RLock()
        self.taxonomy = taxonomy or dict(DEFAULT_TAXONOMY)
        self.categories = sorted(categories or list(self.taxonomy.keys()))
        self.full_categories = sorted(flatten_taxonomy(self.taxonomy))
        self.model_path = Path(model_path)
        self.training_data_path = Path(training_data_path)

        # Feature extractor: character & word n-grams capture app names, domains, and titles
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            max_features=100_000,
            sublinear_tf=True,
        )

        # SGDClassifier with log_loss provides logistic regression with incremental learning
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
        )

        self.is_trained = False
        self.training_samples: List[Dict[str, str]] = []
        self.samples_by_text: Dict[str, Dict[str, Any]] = {}
        # Normalized-key index for cache lookups (volatile numbers/query strings folded away)
        self.samples_by_key: Dict[str, Dict[str, Any]] = {}
        self._unretrained_count: int = 0

        # Load existing state if available
        self._load_existing_state()

    def _load_existing_state(self):
        """Load persistent model or pre-existing training data."""
        if self.model_path.exists():
            try:
                saved = joblib.load(self.model_path)
                self.vectorizer = saved["vectorizer"]
                self.model = saved["model"]
                self.categories = saved.get("categories", self.categories)
                self.is_trained = saved.get("is_trained", True)
                logger.info(f"Loaded existing classifier from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load existing model {self.model_path}: {e}")

        if self.training_data_path.exists():
            try:
                raw_samples = []
                with open(self.training_data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            raw_samples.append(json.loads(line))

                # Sanitize all historic samples, ensure hierarchical format, and deduplicate
                needs_rewrite = False
                samples_by_text: Dict[str, Dict[str, Any]] = {}
                for s in raw_samples:
                    raw_text = s.get("text", "")
                    clean_text = anonymizer.sanitize_text(raw_text)
                    if clean_text != raw_text:
                        needs_rewrite = True

                    cat_raw = s.get("category", "Uncategorized")
                    subcat_raw = s.get("subcategory")
                    full_raw = s.get("full_category")

                    if not full_raw:
                        if " > " in cat_raw:
                            full_raw = cat_raw
                            cat, sub = split_category_path(full_raw)
                        elif subcat_raw:
                            full_raw = f"{cat_raw} > {subcat_raw}"
                            cat, sub = cat_raw, subcat_raw
                        else:
                            cat, sub = cat_raw, "General"
                            full_raw = f"{cat} > {sub}"
                        needs_rewrite = True
                    else:
                        cat, sub = split_category_path(full_raw)

                    sample_dict = {
                        "text": clean_text,
                        "category": cat,
                        "subcategory": sub,
                        "full_category": full_raw,
                        "source": s.get("source", "historic"),
                    }

                    if clean_text in samples_by_text:
                        needs_rewrite = True

                    samples_by_text[clean_text] = sample_dict

                self.samples_by_text = samples_by_text
                self.training_samples = list(samples_by_text.values())
                self._reindex_samples()
                logger.info(f"Loaded {len(self.training_samples)} sanitized training examples.")

                if needs_rewrite:
                    self._rewrite_training_file()
                    logger.info("Rewrote training data file with clean PII and taxonomy format.")

                # If we have training samples, train or retrain
                if not self.is_trained and len(self.training_samples) >= 2:
                    self.retrain_full()
            except Exception as e:
                logger.warning(f"Could not load training data from {self.training_data_path}: {e}")

    def _rewrite_training_file(self):
        """Rewrite the JSONL atomically: a crash mid-write must not truncate the training data."""
        tmp = self.training_data_path.with_suffix(self.training_data_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for s in self.training_samples:
                f.write(json.dumps(s) + "\n")
        os.replace(tmp, self.training_data_path)

    @staticmethod
    def extract_features(event_data: Dict[str, Any]) -> str:
        """Create sanitized textual signature from event payload with zero PII and rich domain/link signals."""
        clean_data = anonymizer.sanitize_for_training(event_data)
        app = str(clean_data.get("app") or "").strip().lower()
        title = str(clean_data.get("title") or "").strip().lower()
        url = str(clean_data.get("url") or "").strip().lower()
        package = str(clean_data.get("package") or "").strip().lower()

        domain = ""
        root_domain = ""
        path_words: List[str] = []

        if url:
            try:
                test_url = url if "://" in url else f"https://{url}"
                parsed = urlparse(test_url)
                netloc = parsed.netloc.lower().split(":")[0]
                domain = netloc

                parts = netloc.split(".")
                if len(parts) >= 2:
                    root_domain = ".".join(parts[-2:])
                else:
                    root_domain = netloc

                for segment in parsed.path.lower().split("/"):
                    seg = segment.strip()
                    if len(seg) >= 3 and not re.match(r"^[0-9a-f\-]+$", seg) and not seg.isdigit():
                        path_words.append(seg)
            except Exception:
                domain = ""
                root_domain = ""

        path_str = " ".join(path_words[:6])
        return f"app:{app} package:{package} domain:{domain} root_domain:{root_domain} path:{path_str} title:{title} url:{url}"

    # ------------------------------------------------------------------
    # Cache keys
    # ------------------------------------------------------------------
    @staticmethod
    def _normalized_key(clean_text: str) -> str:
        """Fold volatile parts (numbers, URL query/fragment) so repeated activity hits the cache."""
        text = clean_text.lower()
        text = re.sub(r"(url:[^\s?#]*)[?#]\S*", r"\1", text)
        text = re.sub(r"\d+", "#", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def cache_key(feature_text: str) -> str:
        return FastClassifier._normalized_key(anonymizer.sanitize_text(feature_text))

    def _reindex_samples(self):
        self.samples_by_key = {}
        for sample in self.training_samples:
            self._index_sample(sample)

    def _index_sample(self, sample: Dict[str, Any]):
        key = self._normalized_key(sample.get("text", ""))
        existing = self.samples_by_key.get(key)
        if existing is None or sample.get("source") == "human_correction" or (
            existing.get("source") != "human_correction"
            and existing.get("source") != sample.get("source")
        ):
            self.samples_by_key[key] = sample

    def is_cached(self, feature_text: str) -> bool:
        return self.cache_key(feature_text) in self.samples_by_key

    @staticmethod
    def _cached_result(sample: Dict[str, Any], source: str) -> Dict[str, Any]:
        full = sample["full_category"]
        cat = sample["category"]
        return {
            "category": cat,
            "subcategory": sample["subcategory"],
            "full_category": full,
            "confidence": 1.0,
            "probs": {full: 1.0, f"category:{cat}": 1.0},
            "source": source,
        }

    # ------------------------------------------------------------------
    # Deterministic rule cascade
    # ------------------------------------------------------------------
    @staticmethod
    def match_rules(feature_text: str) -> Optional[Tuple[str, str]]:
        """Return a high-precision (category, subcategory) from app/domain rules, or None.

        Only strong fields (app, domain, root domain) are used for browsers; titles are
        used for specific applications and terminals, where they are not generic noise.
        """
        f = feature_text.lower()

        if is_email_feature_text(f):
            return ("Communication", "Email")

        fields = _feature_fields(f)
        app = fields.get("app", "")
        domain = fields.get("domain", "")
        root_domain = fields.get("root_domain", "")
        title = fields.get("title", "")
        url = fields.get("url", "")

        # ActivityWatch's own web UI must not be mistaken for a local dev server.
        if "5600" in url and ("127.0.0.1" in url or "localhost" in url):
            return ("System & Admin", "Hardware & System Monitor")

        if app in TERMINAL_APPS:
            match = _seed_match(title)
            if match and match[0] == "System & Admin":
                return match
            return ("Development", "DevOps & Terminal")

        if domain:
            for pattern, cat, sub in _DOMAIN_OVERRIDES:
                if pattern.search(domain):
                    return (cat, sub)
            if domain not in GENERIC_DOMAINS and not domain.startswith(
                ("127.", "192.168.", "10.", "0.0.0.0")
            ):
                match = _seed_match(f"{domain} {root_domain}")
                if match:
                    return match

        if app and app not in GENERIC_HOST_APPS:
            hint = app_knowledge.get_category_hint_exact(app)
            if hint:
                return hint
            match = _seed_match(f"{app} {title}")
            if match:
                return match

        return None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate_proba(
        probs_row: np.ndarray, classes: List[str]
    ) -> Tuple[str, str, str, float, Dict[str, float]]:
        """Aggregate subcategory probabilities into parent category confidence."""
        cat_probs: Dict[str, float] = {}
        for cls_name, p in zip(classes, probs_row):
            c, _ = split_category_path(cls_name)
            cat_probs[c] = cat_probs.get(c, 0.0) + float(p)

        best_cat = max(cat_probs, key=cat_probs.get)
        cat_confidence = float(cat_probs[best_cat])

        sub_candidates = [
            (cls_name, float(p))
            for cls_name, p in zip(classes, probs_row)
            if split_category_path(cls_name)[0] == best_cat
        ]
        best_full = (
            max(sub_candidates, key=lambda x: x[1])[0]
            if sub_candidates
            else f"{best_cat} > General"
        )
        _, best_sub = split_category_path(best_full)

        prob_dict = {cls_name: float(p) for cls_name, p in zip(classes, probs_row)}
        for c, cp in cat_probs.items():
            prob_dict[f"category:{c}"] = round(cp, 4)

        return best_cat, best_sub, best_full, cat_confidence, prob_dict

    @_locked
    def predict_batch_meta(self, feature_texts: List[str]) -> List[Dict[str, Any]]:
        """Classify many feature strings in one vectorizer/model pass.

        Order of precedence: human correction cache -> deterministic rules -> cache -> ML.
        Human corrections always win so a teacher guess can never overwrite them.
        """
        results: List[Optional[Dict[str, Any]]] = [None] * len(feature_texts)
        ml_indices: List[int] = []
        ml_texts: List[str] = []

        for i, raw in enumerate(feature_texts):
            clean = anonymizer.sanitize_text(raw)
            cached = self.samples_by_key.get(self._normalized_key(clean))

            if cached and cached.get("source") == "human_correction":
                results[i] = self._cached_result(cached, "human_cache")
                continue

            rule = self.match_rules(clean)
            if rule:
                cat, sub = rule
                full = f"{cat} > {sub}"
                results[i] = {
                    "category": cat,
                    "subcategory": sub,
                    "full_category": full,
                    "confidence": 0.95,
                    "probs": {full: 0.95, f"category:{cat}": 0.95},
                    "source": "rules",
                }
                continue

            if cached:
                results[i] = self._cached_result(cached, "cache")
                continue

            ml_indices.append(i)
            ml_texts.append(clean)

        if ml_texts and self.is_trained:
            try:
                X = self.vectorizer.transform(ml_texts)
                proba = self.model.predict_proba(X)
                classes = [str(c) for c in self.model.classes_]
                for row, i in enumerate(ml_indices):
                    cat, sub, full, conf, probs = self._aggregate_proba(proba[row], classes)
                    results[i] = {
                        "category": cat,
                        "subcategory": sub,
                        "full_category": full,
                        "confidence": conf,
                        "probs": probs,
                        "source": "fast_classifier",
                    }
            except Exception as e:
                logger.error(f"Batch prediction failed: {e}")
                for i in ml_indices:
                    results[i] = {
                        "category": "Uncategorized",
                        "subcategory": "General",
                        "full_category": "Uncategorized > General",
                        "confidence": 0.0,
                        "probs": {},
                        "source": "fast_classifier",
                    }
        else:
            for i in ml_indices:
                results[i] = {
                    "category": "Uncategorized",
                    "subcategory": "General",
                    "full_category": "Uncategorized > General",
                    "confidence": 0.0,
                    "probs": {},
                    "source": "untrained",
                }

        return results

    def predict_batch(
        self, feature_texts: List[str]
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        return [
            (m["full_category"], m["confidence"], m["probs"])
            for m in self.predict_batch_meta(feature_texts)
        ]

    def predict(self, feature_text: str) -> Tuple[str, float, Dict[str, float]]:
        """Predict full category (Category > Subcategory) and calibrated category confidence score."""
        return self.predict_batch([feature_text])[0]

    def predict_hierarchy(
        self, feature_text: str
    ) -> Tuple[str, str, str, float, Dict[str, float]]:
        """Predict category, subcategory, full_category, and calibrated category confidence score."""
        m = self.predict_batch_meta([feature_text])[0]
        return (m["category"], m["subcategory"], m["full_category"], m["confidence"], m["probs"])

    @_locked
    def add_training_sample(
        self,
        feature_text: str,
        category: str,
        subcategory: Optional[str] = None,
        source: str = "gemma_oracle",
        retrain_now: bool = True,
        retrain_batch_size: int = 10,
    ) -> bool:
        """Record newly labeled sample with full hierarchy, deduplication, and PII sanitization.

        Returns True if a new sample was added or an existing sample updated;
        returns False if the sample already existed with identical classification.
        Automatically retrains every `retrain_batch_size` samples if retrain_now=False.
        """
        clean_text = anonymizer.sanitize_text(feature_text)

        if subcategory:
            cat = category.strip()
            sub = subcategory.strip()
            full_cat = f"{cat} > {sub}"
        elif " > " in category:
            full_cat = category.strip()
            cat, sub = split_category_path(full_cat)
        else:
            cat = category.strip()
            sub = "General"
            full_cat = f"{cat} > {sub}"

        # Deduplicate: check if this exact text already exists (O(1) lookup).
        # A human correction is never overwritten by a teacher/heuristic label.
        if clean_text in self.samples_by_text:
            existing = self.samples_by_text[clean_text]
            existing_source = existing.get("source")

            if existing_source == "human_correction" and source != "human_correction":
                return False

            if existing.get("full_category") == full_cat:
                if source != "human_correction" or existing_source == "human_correction":
                    return False
                # Same label, but now confirmed by a human: mark it protected.
                existing["source"] = source
            else:
                existing["category"] = cat
                existing["subcategory"] = sub
                existing["full_category"] = full_cat
                existing["source"] = source

            self._index_sample(existing)

            try:
                self._rewrite_training_file()
            except Exception as e:
                logger.warning(f"Failed to rewrite {self.training_data_path}: {e}")
        else:
            sample = {
                "text": clean_text,
                "category": cat,
                "subcategory": sub,
                "full_category": full_cat,
                "source": source,
            }
            self.samples_by_text[clean_text] = sample
            self.training_samples.append(sample)
            self._index_sample(sample)

            try:
                with open(self.training_data_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sample) + "\n")
            except Exception as e:
                logger.warning(f"Failed to append sample to {self.training_data_path}: {e}")

        if cat not in self.categories:
            self.categories.append(cat)
            self.categories.sort()

        self._unretrained_count += 1
        unique_classes = set(s["full_category"] for s in self.training_samples)
        can_train = len(unique_classes) >= 2

        if can_train:
            if retrain_now or (retrain_batch_size > 0 and self._unretrained_count >= retrain_batch_size):
                self.retrain_full()

        return True

    @_locked
    def retrain_full(self) -> bool:
        """Full retrain on all accumulated hierarchical training data."""
        if not self.training_samples:
            return False

        texts = [s["text"] for s in self.training_samples]
        labels = [s["full_category"] for s in self.training_samples]

        unique_labels = sorted(list(set(labels)))
        if len(unique_labels) < 2:
            return False

        try:
            X = self.vectorizer.fit_transform(texts)
            self.model.fit(X, labels)
            self.is_trained = True
            self._unretrained_count = 0
            self.save_model()
            logger.info(
                f"Retrained classifier successfully on {len(texts)} samples across {len(unique_labels)} hierarchical classes."
            )
            return True
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return False

    def save_model(self):
        """Serialize model weights and vectorizer."""
        try:
            state = {
                "vectorizer": self.vectorizer,
                "model": self.model,
                "categories": self.categories,
                "is_trained": self.is_trained,
            }
            joblib.dump(state, self.model_path)
            logger.info(f"Classifier saved to {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not save model to {self.model_path}: {e}")

    @_locked
    def get_stats(self) -> Dict[str, Any]:
        """Summary metrics of current classifier state with category and subcategory counts."""
        category_counts: Dict[str, int] = {}
        subcategory_counts: Dict[str, int] = {}

        for s in self.training_samples:
            c = s.get("category", "Uncategorized")
            full = s.get("full_category", f"{c} > General")
            category_counts[c] = category_counts.get(c, 0) + 1
            subcategory_counts[full] = subcategory_counts.get(full, 0) + 1

        return {
            "is_trained": self.is_trained,
            "total_training_samples": len(self.training_samples),
            "cached_samples_count": len(self.samples_by_text),
            "classes_trained": list(self.model.classes_) if self.is_trained else [],
            "category_distribution": category_counts,
            "subcategory_distribution": subcategory_counts,
            "all_configured_categories": self.categories,
            "taxonomy": self.taxonomy,
        }
