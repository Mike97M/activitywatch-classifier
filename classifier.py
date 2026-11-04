"""Fast ML Classifier with Calibrated Confidence Scoring, Hierarchical Taxonomy, and Online Active Learning."""

import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from anonymizer import anonymizer
from config import (
    DEFAULT_CATEGORIES,
    DEFAULT_TAXONOMY,
    MODEL_PATH,
    TRAINING_DATA_PATH,
    flatten_taxonomy,
    split_category_path,
)

logger = logging.getLogger("classifier")


class FastClassifier:
    """Fast, local classifier providing real-time hierarchical predictions and calibrated confidence scores."""

    def __init__(
        self,
        taxonomy: Optional[Dict[str, List[str]]] = None,
        categories: Optional[List[str]] = None,
        model_path: Path = MODEL_PATH,
        training_data_path: Path = TRAINING_DATA_PATH,
    ):
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

                self.training_samples = list(samples_by_text.values())
                logger.info(f"Loaded {len(self.training_samples)} sanitized training examples.")

                if needs_rewrite:
                    with open(self.training_data_path, "w", encoding="utf-8") as f:
                        for s in self.training_samples:
                            f.write(json.dumps(s) + "\n")
                    logger.info("Rewrote training data file with clean PII and taxonomy format.")

                # If we have training samples, train or retrain
                if not self.is_trained and len(self.training_samples) >= 2:
                    self.retrain_full()
            except Exception as e:
                logger.warning(f"Could not load training data from {self.training_data_path}: {e}")

    @staticmethod
    def extract_features(event_data: Dict[str, Any]) -> str:
        """Create sanitized textual signature from event payload with zero PII and rich domain/link signals."""
        clean_data = anonymizer.sanitize_event_data(event_data)
        app = str(clean_data.get("app") or "").strip().lower()
        title = str(clean_data.get("title") or "").strip().lower()
        url = str(clean_data.get("url") or "").strip().lower()

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
        return f"app:{app} domain:{domain} root_domain:{root_domain} path:{path_str} title:{title} url:{url}"

    def predict(self, feature_text: str) -> Tuple[str, float, Dict[str, float]]:
        """Predict full category (Category > Subcategory) and confidence score."""
        if not self.is_trained:
            return ("Uncategorized > General", 0.0, {})

        try:
            X = self.vectorizer.transform([feature_text])
            probs = self.model.predict_proba(X)[0]
            classes = self.model.classes_

            best_idx = int(np.argmax(probs))
            predicted_full = str(classes[best_idx])
            confidence = float(probs[best_idx])

            prob_dict = {str(c): float(p) for c, p in zip(classes, probs)}
            return (predicted_full, confidence, prob_dict)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return ("Uncategorized > General", 0.0, {})

    def predict_hierarchy(
        self, feature_text: str
    ) -> Tuple[str, str, str, float, Dict[str, float]]:
        """Predict category, subcategory, full_category, and confidence score."""
        full_cat, conf, probs = self.predict(feature_text)
        cat, subcat = split_category_path(full_cat)
        return (cat, subcat, full_cat, conf, probs)

    def add_training_sample(
        self,
        feature_text: str,
        category: str,
        subcategory: Optional[str] = None,
        source: str = "gemma_oracle",
        retrain_now: bool = True,
    ) -> bool:
        """Record newly labeled sample with full hierarchy, deduplication, and PII sanitization.

        Returns True if a new sample was added or an existing sample updated;
        returns False if the sample already existed with identical classification.
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

        # Deduplicate: check if this text already exists
        existing_idx = None
        for i, s in enumerate(self.training_samples):
            if s.get("text") == clean_text:
                existing_idx = i
                break

        if existing_idx is not None:
            existing = self.training_samples[existing_idx]
            if existing.get("full_category") == full_cat:
                return False

            # Existing entry has different category -> update in place (e.g. human correction)
            existing["category"] = cat
            existing["subcategory"] = sub
            existing["full_category"] = full_cat
            existing["source"] = source

            try:
                with open(self.training_data_path, "w", encoding="utf-8") as f:
                    for s in self.training_samples:
                        f.write(json.dumps(s) + "\n")
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
            self.training_samples.append(sample)

            try:
                with open(self.training_data_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sample) + "\n")
            except Exception as e:
                logger.warning(f"Failed to append sample to {self.training_data_path}: {e}")

        if cat not in self.categories:
            self.categories.append(cat)
            self.categories.sort()

        if retrain_now:
            unique_classes = set(s["full_category"] for s in self.training_samples)
            if len(unique_classes) >= 2:
                self.retrain_full()

        return True

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
            "classes_trained": list(self.model.classes_) if self.is_trained else [],
            "category_distribution": category_counts,
            "subcategory_distribution": subcategory_counts,
            "all_configured_categories": self.categories,
            "taxonomy": self.taxonomy,
        }
