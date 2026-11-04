"""Pipeline coordinating ActivityWatch buckets, AFK filtering, FastClassifier, and Gemma Oracle with hierarchical subcategories."""

from datetime import timedelta
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from afk_filter import filter_events_by_afk, parse_iso_timestamp, to_iso_timestamp
from anonymizer import anonymizer
from aw_client import AWClient
from classifier import FastClassifier
from config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TARGET_BUCKET_ID,
    TRAINING_DATA_PATH,
    split_category_path,
)
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
        self.gemma_oracle = gemma_oracle or GemmaOracle(taxonomy=self.classifier.taxonomy)
        self.confidence_threshold = confidence_threshold

        self.last_run_stats: Dict[str, Any] = {}

    def run(
        self,
        source_bucket_ids: List[str],
        target_bucket_id: str = DEFAULT_TARGET_BUCKET_ID,
        limit_per_bucket: int = 200,
        confidence_threshold: Optional[float] = None,
        filter_afk: bool = True,
        afk_bucket_id: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        collect_training_data: bool = True,
        auto_commit_git: bool = False,
    ) -> Dict[str, Any]:
        """Clone events from source buckets, filter out AFK intervals, classify them into category and subcategory, and save into target bucket."""
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

        total_raw_events = 0
        total_afk_filtered = 0
        total_processed = 0
        fast_calls = 0
        gemma_calls = 0
        new_training_samples = 0
        afk_buckets_used: Set[str] = set()
        category_distribution: Dict[str, int] = {}
        subcategory_distribution: Dict[str, int] = {}
        classified_events_to_insert: List[Dict[str, Any]] = []
        recent_preview: List[Dict[str, Any]] = []

        for source_id in source_bucket_ids:
            try:
                events = self.aw_client.get_events(source_id, limit=limit_per_bucket)
            except Exception as e:
                logger.error(f"Failed to fetch events from {source_id}: {e}")
                continue

            raw_event_count = len(events)
            total_raw_events += raw_event_count

            # AFK Filtering: account for AFK bucket and only consider items when user wasn't AFK
            if filter_afk and events:
                matched_afk = afk_bucket_id or self.aw_client.find_afk_bucket_for_source(source_id)
                if matched_afk:
                    try:
                        min_time = min(parse_iso_timestamp(ev["timestamp"]) for ev in events)
                        max_time = max(
                            parse_iso_timestamp(ev["timestamp"])
                            + timedelta(seconds=float(ev.get("duration", 0.0)))
                            for ev in events
                        )
                        buf_start = to_iso_timestamp(min_time - timedelta(minutes=15))
                        buf_end = to_iso_timestamp(max_time + timedelta(minutes=15))

                        afk_events = self.aw_client.get_events(
                            matched_afk, start=buf_start, end=buf_end, limit=0
                        )
                        if afk_events:
                            events = filter_events_by_afk(events, afk_events)
                            afk_buckets_used.add(matched_afk)
                            dropped = raw_event_count - len(events)
                            if dropped > 0:
                                total_afk_filtered += dropped
                                logger.info(
                                    f"AFK Guard: filtered out {dropped} idle/AFK events from {source_id} using {matched_afk}"
                                )
                        else:
                            logger.warning(
                                f"No AFK events found in {matched_afk} between {buf_start} and {buf_end}. Keeping source events."
                            )
                    except Exception as e:
                        logger.warning(
                            f"AFK filtering failed for {source_id}: {e}. Proceeding with source events."
                        )
                else:
                    logger.info(f"No matching AFK bucket found for {source_id}. Proceeding with raw events.")

            for event in events:
                raw_data = event.get("data", {})
                clean_data = anonymizer.sanitize_event_data(raw_data)
                app = clean_data.get("app", "")
                
                # If app is empty (common in web watcher buckets), infer from bucket ID
                if not app and source_id:
                    s_lower = source_id.lower()
                    if "firefox" in s_lower:
                        app = "firefox"
                    elif "brave" in s_lower:
                        app = "brave"
                    elif "chromium" in s_lower or "chrome" in s_lower:
                        app = "chromium"
                    elif "edge" in s_lower:
                        app = "edge"
                    elif "web" in s_lower:
                        app = "browser"
                    if app:
                        clean_data["app"] = app

                title = clean_data.get("title", "")
                url = clean_data.get("url", "")

                feature_text = self.classifier.extract_features(clean_data)

                # Attempt fast hierarchical prediction
                pred_cat, pred_subcat, pred_full, confidence, prob_dict = (
                    self.classifier.predict_hierarchy(feature_text)
                )

                # If confidence is lower than threshold, call Gemma
                if confidence < threshold:
                    gemma_res = self.gemma_oracle.classify_activity(
                        app=app,
                        title=title,
                        url=url,
                        taxonomy=self.classifier.taxonomy,
                    )
                    chosen_category = gemma_res.get("category", "Uncategorized")
                    chosen_subcategory = gemma_res.get("subcategory", "General Activity")
                    chosen_full = gemma_res.get(
                        "full_category", f"{chosen_category} > {chosen_subcategory}"
                    )
                    chosen_confidence = float(gemma_res.get("confidence", 0.95))
                    classified_by = f"gemma_e2b ({gemma_res.get('backend', 'oracle')})"
                    reasoning = gemma_res.get("reasoning", "")
                    gemma_calls += 1

                    # Add new training sample if enabled (deferred retrain for batch speed)
                    if collect_training_data:
                        added = self.classifier.add_training_sample(
                            feature_text=feature_text,
                            category=chosen_category,
                            subcategory=chosen_subcategory,
                            source="gemma_e2b",
                            retrain_now=False,
                        )
                        if added:
                            new_training_samples += 1
                else:
                    chosen_category = pred_cat
                    chosen_subcategory = pred_subcat
                    chosen_full = pred_full
                    chosen_confidence = confidence
                    classified_by = "fast_classifier"
                    reasoning = f"High confidence prediction ({confidence:.2f})"
                    fast_calls += 1

                    # Collect novel confident activities to continually expand training dataset
                    if collect_training_data and confidence >= 0.70:
                        added = self.classifier.add_training_sample(
                            feature_text=feature_text,
                            category=chosen_category,
                            subcategory=chosen_subcategory,
                            source="fast_classifier",
                            retrain_now=False,
                        )
                        if added:
                            new_training_samples += 1

                category_distribution[chosen_category] = (
                    category_distribution.get(chosen_category, 0) + 1
                )
                subcategory_distribution[chosen_full] = (
                    subcategory_distribution.get(chosen_full, 0) + 1
                )
                total_processed += 1

                # Construct the enriched non-destructive event for the target bucket
                cloned_data = dict(clean_data)
                cloned_data["category"] = chosen_category
                cloned_data["subcategory"] = chosen_subcategory
                cloned_data["full_category"] = chosen_full
                cloned_data["category_hierarchy"] = [chosen_category, chosen_subcategory]
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
                            "subcategory": chosen_subcategory,
                            "full_category": chosen_full,
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
                            "current_subcategory": chosen_subcategory,
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

            # Synchronize to ActivityWatch host 'classified' so events appear in the Activity tab
            try:
                self.aw_client.sync_classified_host_bucket(classified_events_to_insert, host="classified")
            except Exception as e:
                logger.warning(f"Failed to sync classified activities to Activity tab host: {e}")

            # Synchronize taxonomy & rules into ActivityWatch settings so Activity tab categorizes them
            try:
                from category_sync import sync_taxonomy_to_activitywatch
                sync_taxonomy_to_activitywatch(self.aw_client, self.classifier)
            except Exception as e:
                logger.warning(f"Failed to auto-sync category rules to ActivityWatch: {e}")

        # Retrain fast classifier if new samples were added during batch
        if new_training_samples > 0:
            logger.info(
                f"Retraining fast classifier with {new_training_samples} new unique training samples..."
            )
            self.classifier.retrain_full()

        # Git dataset commit
        git_commit_res = None
        git_status_res = None
        try:
            import git_sync
            if (
                auto_commit_git
                and new_training_samples > 0
                and Path(self.classifier.training_data_path).resolve() == Path(TRAINING_DATA_PATH).resolve()
            ):
                git_commit_res = git_sync.commit_training_data(
                    message=f"data: auto-add {new_training_samples} training samples via classifier pipeline"
                )
            git_status_res = git_sync.get_training_data_status()
        except Exception as e:
            logger.warning(f"Git dataset sync failed: {e}")

        stats = {
            "target_bucket_id": target_bucket_id,
            "source_buckets": source_bucket_ids,
            "afk_filtering_enabled": filter_afk,
            "afk_buckets_used": sorted(list(afk_buckets_used)),
            "total_raw_events_fetched": total_raw_events,
            "afk_events_filtered_out": total_afk_filtered,
            "total_events_processed": total_processed,
            "events_saved_to_new_bucket": inserted_count,
            "fast_classifier_calls": fast_calls,
            "gemma_oracle_calls": gemma_calls,
            "confidence_threshold": threshold,
            "new_training_samples_added": new_training_samples,
            "git_commit": git_commit_res,
            "git_status": git_status_res,
            "category_distribution": category_distribution,
            "subcategory_distribution": subcategory_distribution,
            "recent_preview": recent_preview,
            "classifier_stats": self.classifier.get_stats(),
        }
        self.last_run_stats = stats
        return stats
