"""Fast ML Classifier with Calibrated Confidence Scoring and Online Active Learning."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from config import DEFAULT_CATEGORIES, MODEL_PATH, TRAINING_DATA_PATH

logger = logging.getLogger("classifier")


class FastClassifier:
    """Fast, local classifier providing real-time predictions and calibrated confidence scores."""

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        model_path: Path = MODEL_PATH,
        training_data_path: Path = TRAINING_DATA_PATH,
    ):
        self.categories = sorted(categories or list(DEFAULT_CATEGORIES))
        self.model_path = Path(model_path)
        self.training_data_path = Path(training_data_path)

        # Feature extractor: character & word n-grams capture app names, domains, and titles
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True,
        )

        # SGDClassifier with log_loss provides logistic regression with incremental partial_fit
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

        # Attempt to load saved state
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
                with open(self.training_data_path, "r", encoding="utf-8") as f:
                    self.training_samples = [json.loads(line) for line in f if line.strip()]
                logger.info(f"Loaded {len(self.training_samples)} historic training examples.")
                # If we have training samples but model wasn't trained yet, fit it now
                if not self.is_trained and len(self.training_samples) >= 2:
                    self.retrain_full()
            except Exception as e:
                logger.warning(f"Could not load training data from {self.training_data_path}: {e}")

    @staticmethod
    def extract_features(event_data: Dict[str, Any]) -> str:
        """Create clean textual signature from event payload."""
        app = str(event_data.get("app") or "").strip().lower()
        title = str(event_data.get("title") or "").strip().lower()
        url = str(event_data.get("url") or "").strip().lower()

        # Extract domain from url if present
        domain = ""
        if url.startswith("http://") or url.startswith("https://"):
            parts = url.split("/")
            if len(parts) > 2:
                domain = parts[2]

        return f"app:{app} domain:{domain} title:{title} url:{url}"

    def predict(self, feature_text: str) -> Tuple[str, float, Dict[str, float]]:
        """Predict category and return calibrated confidence score (0.0 to 1.0).

        Returns: (predicted_category, confidence_score, all_class_probabilities)
        """
        if not self.is_trained:
            # Model has not seen enough data yet
            return ("Uncategorized", 0.0, {})

        try:
            X = self.vectorizer.transform([feature_text])
            probs = self.model.predict_proba(X)[0]
            classes = self.model.classes_

            best_idx = int(np.argmax(probs))
            predicted_category = str(classes[best_idx])
            confidence = float(probs[best_idx])

            prob_dict = {str(c): float(p) for c, p in zip(classes, probs)}
            return (predicted_category, confidence, prob_dict)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return ("Uncategorized", 0.0, {})

    def add_training_sample(
        self,
        feature_text: str,
        category: str,
        source: str = "gemma_oracle",
        retrain_now: bool = True,
    ):
        """Record newly labeled sample from Gemma and update model."""
        sample = {
            "text": feature_text,
            "category": category,
            "source": source,
        }
        self.training_samples.append(sample)

        # Append to jsonl
        try:
            with open(self.training_data_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")
        except Exception as e:
            logger.warning(f"Failed to append sample to {self.training_data_path}: {e}")

        # Update categories if new category encountered
        if category not in self.categories:
            self.categories.append(category)
            self.categories.sort()

        if retrain_now:
            # If we already have multiple classes represented, incremental fit or retrain
            unique_classes = set(s["category"] for s in self.training_samples)
            if len(unique_classes) >= 2:
                self.retrain_full()

    def retrain_full(self) -> bool:
        """Full retrain on all accumulated training data."""
        if not self.training_samples:
            return False

        texts = [s["text"] for s in self.training_samples]
        labels = [s["category"] for s in self.training_samples]

        unique_labels = sorted(list(set(labels)))
        if len(unique_labels) < 2:
            # Cannot train a classifier with only 1 class
            return False

        try:
            # Refit vectorizer and model
            X = self.vectorizer.fit_transform(texts)
            self.model.fit(X, labels)
            self.is_trained = True
            self.save_model()
            logger.info(
                f"Retrained classifier successfully on {len(texts)} samples across {len(unique_labels)} classes."
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
        """Summary metrics of current classifier state."""
        category_counts: Dict[str, int] = {}
        for s in self.training_samples:
            c = s["category"]
            category_counts[c] = category_counts.get(c, 0) + 1

        return {
            "is_trained": self.is_trained,
            "total_training_samples": len(self.training_samples),
            "classes_trained": list(self.model.classes_) if self.is_trained else [],
            "category_distribution": category_counts,
            "all_configured_categories": self.categories,
        }
