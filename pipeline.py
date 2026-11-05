"""Pipeline coordinating ActivityWatch buckets, AFK filtering, FastClassifier, and Gemma Oracle with hierarchical subcategories."""

from datetime import timedelta
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from afk_filter import (
    filter_events_by_afk,
    parse_iso_timestamp,
    resolve_time_range,
    to_iso_timestamp,
)
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
        limit_per_bucket: int = 150,
        time_range: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        filter_afk: bool = True,
        include_audible: bool = True,
        afk_bucket_id: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        collect_training_data: bool = True,
        auto_commit_git: bool = False,
    ) -> Dict[str, Any]:
        """Clone events from source buckets by count or time range, filter AFK intervals, and classify."""
        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.confidence_threshold
        )

        # Resolve time range bounds if requested (e.g. today, 24h, 7d, 30d, all)
        start_iso, end_iso = resolve_time_range(
            time_range=time_range,
            start_time=start_time,
            end_time=end_time,
        )

        # If a time range was requested and limit was default 150, fetch all events in range (limit=0)
        fetch_limit = limit_per_bucket
        if (
            start_iso
            or end_iso
            or (time_range and time_range.lower() in ("today", "24h", "last_day", "7d", "last_week", "30d", "last_month", "all"))
        ) and limit_per_bucket == 150:
            fetch_limit = 0

        # Step 1: Ensure target bucket exists
        if progress_callback:
            progress_callback({
                "step": "init",
                "message": f"Ensuring target bucket '{target_bucket_id}' exists...",
                "percent": 5.0,
            })
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
        audible_buckets_used: Set[str] = set()
        web_events_cache: Dict[str, List[Dict[str, Any]]] = {}
        category_distribution: Dict[str, int] = {}
        subcategory_distribution: Dict[str, int] = {}
        classified_events_to_insert: List[Dict[str, Any]] = []
        recent_preview: List[Dict[str, Any]] = []

        # Step 2: Fetch and filter events across all source buckets
        all_events_to_classify: List[Tuple[str, Dict[str, Any]]] = []

        for idx, source_id in enumerate(source_bucket_ids):
            if progress_callback:
                progress_callback({
                    "step": "fetching",
                    "message": f"Fetching activities from {source_id} ({idx+1}/{len(source_bucket_ids)})...",
                    "percent": round(5.0 + (idx / max(1, len(source_bucket_ids))) * 15.0, 1),
                })
            try:
                events = self.aw_client.get_events(
                    source_id,
                    limit=fetch_limit,
                    start=start_iso,
                    end=end_iso,
                )
            except Exception as e:
                logger.error(f"Failed to fetch events from {source_id}: {e}")
                continue

            raw_event_count = len(events)
            total_raw_events += raw_event_count

            # AFK Filtering: account for AFK bucket and only consider items when user wasn't AFK
            if filter_afk and events:
                if progress_callback:
                    progress_callback({
                        "step": "filtering_afk",
                        "message": f"Filtering idle/AFK periods for {source_id}...",
                        "percent": round(5.0 + ((idx + 0.5) / max(1, len(source_bucket_ids))) * 15.0, 1),
                    })
                # Per-device AFK resolution:
                # If afk_bucket_id is provided, ensure it belongs to the same device/hostname,
                # otherwise resolve strictly for this device to prevent phone events from being filtered by laptop AFK.
                if afk_bucket_id and afk_bucket_id.lower() not in ("auto", "none", ""):
                    source_info = self.aw_client.list_buckets().get(source_id, {})
                    afk_info = self.aw_client.list_buckets().get(afk_bucket_id, {})
                    s_host = source_info.get("hostname", "")
                    a_host = afk_info.get("hostname", "")
                    if s_host and a_host and s_host != a_host:
                        matched_afk = self.aw_client.find_afk_bucket_for_source(source_id)
                    else:
                        matched_afk = afk_bucket_id
                else:
                    matched_afk = self.aw_client.find_afk_bucket_for_source(source_id)
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

                        # Audible tab consideration: tabs playing audio (except background music) count as active
                        audible_candidate_events: List[Dict[str, Any]] = []
                        if include_audible:
                            web_buckets = self.aw_client.find_web_buckets_for_source(source_id)
                            for wb in web_buckets:
                                if wb not in web_events_cache:
                                    try:
                                        wb_evs = self.aw_client.get_events(
                                            wb, start=buf_start, end=buf_end, limit=0
                                        )
                                        web_events_cache[wb] = wb_evs
                                    except Exception as e:
                                        logger.warning(f"Could not fetch web events from {wb}: {e}")
                                        web_events_cache[wb] = []
                                audible_candidate_events.extend(web_events_cache[wb])
                                audible_buckets_used.add(wb)

                        if afk_events or audible_candidate_events:
                            events = filter_events_by_afk(
                                events,
                                afk_events,
                                audible_events=audible_candidate_events,
                                include_audible=include_audible,
                            )
                            afk_buckets_used.add(matched_afk)
                            dropped = raw_event_count - len(events)
                            if dropped > 0:
                                total_afk_filtered += dropped
                                logger.info(
                                    f"AFK Guard: filtered out {dropped} idle/AFK events from {source_id} using {matched_afk} (include_audible={include_audible})"
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
                all_events_to_classify.append((source_id, event))

        total_to_classify = len(all_events_to_classify)
        logger.info(f"Total active events to classify across {len(source_bucket_ids)} buckets: {total_to_classify}")

        # Step 3: Classify events
        # Pass 1: FastClassifier inference & grouping activities needing Gemma
        event_records: List[Dict[str, Any]] = []
        events_needing_gemma: Dict[Tuple[str, str, str], List[int]] = {}
        gemma_sig_features: Dict[Tuple[str, str, str], str] = {}

        for idx, (source_id, event) in enumerate(all_events_to_classify):
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

            # Fast hierarchical prediction
            pred_cat, pred_subcat, pred_full, confidence, prob_dict = (
                self.classifier.predict_hierarchy(feature_text)
            )

            rec = {
                "source_id": source_id,
                "event": event,
                "clean_data": clean_data,
                "app": app,
                "title": title,
                "url": url,
                "feature_text": feature_text,
                "chosen_category": pred_cat,
                "chosen_subcategory": pred_subcat,
                "chosen_full": pred_full,
                "chosen_confidence": confidence,
                "classified_by": "fast_classifier",
                "reasoning": f"High confidence prediction ({confidence:.2f})",
                "needs_gemma": False,
            }

            if confidence < threshold:
                rec["needs_gemma"] = True
                sig = (app, title, url)
                events_needing_gemma.setdefault(sig, []).append(idx)
                gemma_sig_features[sig] = feature_text

            event_records.append(rec)

        # Pass 2: Batch Gemma multi-stream parallel decoding for unique signatures
        unique_sigs = list(events_needing_gemma.keys())
        if unique_sigs:
            total_gemma_events = sum(len(indices) for indices in events_needing_gemma.values())
            logger.info(
                f"Batching {len(unique_sigs)} unique activity signatures across {total_gemma_events} events "
                f"to Gemma Oracle using multi-stream parallel decoding..."
            )
            activities_to_batch = [
                {"id": i, "app": sig[0], "title": sig[1], "url": sig[2]}
                for i, sig in enumerate(unique_sigs)
            ]

            if progress_callback:
                progress_callback({
                    "step": "gemma_batch",
                    "message": f"Evaluating {len(unique_sigs)} unique activities with Gemma multi-stream parallel decoding...",
                    "percent": 30.0,
                })

            gemma_results = self.gemma_oracle.classify_batch(
                activities_to_batch, taxonomy=self.classifier.taxonomy, max_streams=8
            )

            for sig, gemma_res in zip(unique_sigs, gemma_results):
                cat = gemma_res.get("category", "Uncategorized")
                sub = gemma_res.get("subcategory", "General Activity")
                full = gemma_res.get("full_category", f"{cat} > {sub}")
                conf = float(gemma_res.get("confidence", 0.95))
                backend_info = gemma_res.get("backend", "oracle")
                by = f"gemma_e2b ({backend_info})"
                reas = gemma_res.get("reasoning", "")
                
                feat_text = gemma_sig_features[sig]
                # Collect training samples ONLY from Gemma Oracle (never fast_classifier)
                if collect_training_data:
                    added = self.classifier.add_training_sample(
                        feature_text=feat_text,
                        category=cat,
                        subcategory=sub,
                        source="gemma_e2b",
                        retrain_now=False,
                    )
                    if added:
                        new_training_samples += 1

                # Apply to all events sharing this exact signature
                event_indices = events_needing_gemma[sig]
                gemma_calls += len(event_indices)
                for e_idx in event_indices:
                    r = event_records[e_idx]
                    r["chosen_category"] = cat
                    r["chosen_subcategory"] = sub
                    r["chosen_full"] = full
                    r["chosen_confidence"] = conf
                    r["classified_by"] = by
                    r["reasoning"] = reas

        # Pass 3: Assemble results and statistics
        for idx, rec in enumerate(event_records):
            if not rec["needs_gemma"]:
                fast_calls += 1

            cat = rec["chosen_category"]
            full = rec["chosen_full"]
            category_distribution[cat] = category_distribution.get(cat, 0) + 1
            subcategory_distribution[full] = subcategory_distribution.get(full, 0) + 1
            total_processed += 1

            cloned_data = dict(rec["clean_data"])
            cloned_data["category"] = cat
            cloned_data["subcategory"] = rec["chosen_subcategory"]
            cloned_data["full_category"] = full
            cloned_data["category_hierarchy"] = [cat, rec["chosen_subcategory"]]
            cloned_data["confidence"] = round(rec["chosen_confidence"], 4)
            cloned_data["classified_by"] = rec["classified_by"]
            cloned_data["reasoning"] = rec["reasoning"]
            cloned_data["source_bucket"] = rec["source_id"]

            new_event = {
                "timestamp": rec["event"].get("timestamp"),
                "duration": rec["event"].get("duration", 0.0),
                "data": cloned_data,
            }
            classified_events_to_insert.append(new_event)

            if len(recent_preview) < 50:
                recent_preview.append(
                    {
                        "timestamp": rec["event"].get("timestamp"),
                        "duration": rec["event"].get("duration", 0.0),
                        "app": rec["app"],
                        "title": rec["title"],
                        "url": rec["url"],
                        "category": cat,
                        "subcategory": rec["chosen_subcategory"],
                        "full_category": full,
                        "confidence": round(rec["chosen_confidence"], 3),
                        "classified_by": rec["classified_by"],
                        "reasoning": rec["reasoning"],
                        "source_bucket": rec["source_id"],
                    }
                )

            if progress_callback and (total_processed % 20 == 0 or total_processed == total_to_classify):
                pct = 20.0 + (float(total_processed) / max(1, total_to_classify)) * 60.0
                progress_callback(
                    {
                        "step": "classifying",
                        "current": total_processed,
                        "total": total_to_classify,
                        "percent": round(pct, 1),
                        "message": f"Processed {total_processed}/{total_to_classify} activities ({fast_calls} fast, {gemma_calls} Gemma)...",
                        "processed": total_processed,
                        "fast_calls": fast_calls,
                        "gemma_calls": gemma_calls,
                    }
                )

        # Batch write to ActivityWatch target bucket
        inserted_count = 0
        if classified_events_to_insert:
            if progress_callback:
                progress_callback({
                    "step": "inserting",
                    "current": total_processed,
                    "total": total_to_classify,
                    "percent": 82.0,
                    "message": f"Saving {len(classified_events_to_insert)} categorized events to ActivityWatch...",
                })
            try:
                res = self.aw_client.insert_events(target_bucket_id, classified_events_to_insert)
                inserted_count = len(res)
            except Exception as e:
                logger.error(f"Failed inserting into target bucket {target_bucket_id}: {e}")

            # Synchronize to ActivityWatch host 'classified' so events appear in the Activity tab
            if progress_callback:
                progress_callback({
                    "step": "syncing",
                    "current": total_processed,
                    "total": total_to_classify,
                    "percent": 87.0,
                    "message": "Synchronizing activities to ActivityWatch Activity tab...",
                })
            try:
                self.aw_client.sync_classified_host_bucket(classified_events_to_insert, host="classified")
            except Exception as e:
                logger.warning(f"Failed to sync classified activities to Activity tab host: {e}")

            # Synchronize taxonomy & rules into ActivityWatch settings so Activity tab categorizes them
            if progress_callback:
                progress_callback({
                    "step": "syncing_rules",
                    "current": total_processed,
                    "total": total_to_classify,
                    "percent": 91.0,
                    "message": "Syncing category rules into ActivityWatch settings...",
                })
            try:
                from category_sync import sync_taxonomy_to_activitywatch
                sync_taxonomy_to_activitywatch(self.aw_client, self.classifier)
            except Exception as e:
                logger.warning(f"Failed to auto-sync category rules to ActivityWatch: {e}")

        # Retrain fast classifier if new samples were added during batch
        if new_training_samples > 0:
            if progress_callback:
                progress_callback({
                    "step": "retraining",
                    "current": total_processed,
                    "total": total_to_classify,
                    "percent": 95.0,
                    "message": f"Retraining fast classifier with {new_training_samples} new unique samples...",
                })
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
            "time_range": time_range or "count",
            "start_time": start_iso,
            "end_time": end_iso,
            "limit_per_bucket": fetch_limit,
            "afk_filtering_enabled": filter_afk,
            "afk_buckets_used": sorted(list(afk_buckets_used)),
            "audible_filtering_enabled": include_audible,
            "audible_buckets_used": sorted(list(audible_buckets_used)),
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

        if progress_callback:
            progress_callback({
                "step": "completed",
                "current": total_processed,
                "total": total_to_classify,
                "percent": 100.0,
                "message": f"Successfully classified {total_processed} activities!",
                "results": stats,
            })

        return stats
