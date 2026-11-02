"""Pipeline coordinating ActivityWatch buckets, FastClassifier, and Gemma Oracle."""

import logging
from typing import Any, Callable, Dict, List, Optional
from aw_client import AWClient
from classifier import FastClassifier
from config import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_TARGET_BUCKET_ID
from gemma_oracle import GemmaOracle

logger = logging.getLogger("pipeline")


class ClassificationPipeline:
    def __init__(
        self,
        aw_client: Optional[AWClient] = None,
        classifier: Optional[FastClassifier] = None,
        gemma_oracle: Optional[GemmaOracle] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self.aw_client = aw_client or AWClient()
        self.classifier = classifier or FastClassifier()
        self.gemma_oracle = gemma_oracle or GemmaOracle(categories=self.classifier.categories)
        self.confidence_threshold = confidence_threshold

        self.last_run_stats: Dict[str, Any] = {}

    def run(
        self,
        source_bucket_ids: List[str],
        target_bucket_id: str = DEFAULT_TARGET_BUCKET_ID,
        limit_per_bucket: int = 200,
        confidence_threshold: Optional[float] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Clone events from source buckets, classify them, and save into target bucket."""
        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.confidence_threshold
        )

        # Step 1: Ensure target bucket exists
        self.aw_client.create_bucket(
            bucket_id=target_bucket_id,
            bucket_type="app.classified.activities",
            client_name="aw-gemma-classifier",
        )

        total_processed = 0
        fast_calls = 0
        gemma_calls = 0
        category_distribution: Dict[str, int] = {}
        classified_events_to_insert: List[Dict[str, Any]] = []
        recent_preview: List[Dict[str, Any]] = []

        for source_id in source_bucket_ids:
            try:
                events = self.aw_client.get_events(source_id, limit=limit_per_bucket)
            except Exception as e:
                logger.error(f"Failed to fetch events from {source_id}: {e}")
                continue

            for event in events:
                data = event.get("data", {})
                app = data.get("app", "")
                title = data.get("title", "")
                url = data.get("url", "")

                feature_text = self.classifier.extract_features(data)

                # Attempt fast prediction
                pred_category, confidence, prob_dict = self.classifier.predict(feature_text)

                # If confidence is lower than threshold, call Gemma
                if confidence < threshold:
                    gemma_res = self.gemma_oracle.classify_activity(
                        app=app,
                        title=title,
                        url=url,
                        categories=self.classifier.categories,
                    )
                    chosen_category = gemma_res.get("category", "Uncategorized")
                    chosen_confidence = float(gemma_res.get("confidence", 0.95))
                    classified_by = f"gemma_e2b ({gemma_res.get('backend', 'oracle')})"
                    reasoning = gemma_res.get("reasoning", "")
                    gemma_calls += 1

                    # Add new training sample to improve the classifier model
                    self.classifier.add_training_sample(
                        feature_text=feature_text,
                        category=chosen_category,
                        source="gemma_e2b",
                        retrain_now=True,
                    )
                else:
                    chosen_category = pred_category
                    chosen_confidence = confidence
                    classified_by = "fast_classifier"
                    reasoning = f"High confidence prediction ({confidence:.2f})"
                    fast_calls += 1

                category_distribution[chosen_category] = (
                    category_distribution.get(chosen_category, 0) + 1
                )
                total_processed += 1

                # Construct the enriched non-destructive event for the target bucket
                cloned_data = dict(data)
                cloned_data["category"] = chosen_category
                cloned_data["confidence"] = round(chosen_confidence, 4)
                cloned_data["classified_by"] = classified_by
                cloned_data["reasoning"] = reasoning
                cloned_data["source_bucket"] = source_id

                new_event = {
                    "timestamp": event.get("timestamp"),
                    "duration": event.get("duration", 0.0),
                    "data": cloned_data,
                }
                classified_events_to_insert.append(new_event)

                if len(recent_preview) < 50:
                    recent_preview.append(
                        {
                            "timestamp": event.get("timestamp"),
                            "duration": event.get("duration", 0.0),
                            "app": app,
                            "title": title,
                            "url": url,
                            "category": chosen_category,
                            "confidence": round(chosen_confidence, 3),
                            "classified_by": classified_by,
                            "reasoning": reasoning,
                            "source_bucket": source_id,
                        }
                    )

                if progress_callback and total_processed % 10 == 0:
                    progress_callback(
                        {
                            "processed": total_processed,
                            "fast_calls": fast_calls,
                            "gemma_calls": gemma_calls,
                            "current_category": chosen_category,
                        }
                    )

        # Batch write to ActivityWatch target bucket
        inserted_count = 0
        if classified_events_to_insert:
            try:
                res = self.aw_client.insert_events(target_bucket_id, classified_events_to_insert)
                inserted_count = len(res)
            except Exception as e:
                logger.error(f"Failed inserting into target bucket {target_bucket_id}: {e}")

        stats = {
            "target_bucket_id": target_bucket_id,
            "source_buckets": source_bucket_ids,
            "total_events_processed": total_processed,
            "events_saved_to_new_bucket": inserted_count,
            "fast_classifier_calls": fast_calls,
            "gemma_oracle_calls": gemma_calls,
            "confidence_threshold": threshold,
            "category_distribution": category_distribution,
            "recent_preview": recent_preview,
            "classifier_stats": self.classifier.get_stats(),
        }
        self.last_run_stats = stats
        return stats
