"""Unit and integration tests for ClassificationPipeline."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from classifier import FastClassifier
from gemma_oracle import GemmaOracle
from pipeline import ClassificationPipeline


def test_pipeline_active_learning_and_cloning():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.joblib"
        data_path = Path(tmpdir) / "data.jsonl"

        mock_aw = MagicMock()
        mock_aw.find_afk_bucket_for_source.return_value = None
        mock_aw.create_bucket.return_value = {"id": "test-target", "created": True}
        mock_aw.get_events.return_value = [
            {
                "timestamp": "2026-11-05T12:00:00Z",
                "duration": 60.0,
                "data": {"app": "customtool", "title": "Quarterly planning board"},
            },
            {
                "timestamp": "2026-11-05T12:01:00Z",
                "duration": 120.0,
                "data": {"app": "customtool", "title": "Budget spreadsheet view"},
            },
        ]
        mock_aw.insert_events.side_effect = lambda bucket_id, evs: evs

        classifier = FastClassifier(model_path=model_path, training_data_path=data_path)
        oracle = MagicMock()
        oracle.classify_batch.return_value = [
            {
                "category": "Development",
                "subcategory": "Coding & IDE",
                "full_category": "Development > Coding & IDE",
                "confidence": 0.95,
                "reasoning": "stub",
                "backend": "test",
            },
            {
                "category": "Communication",
                "subcategory": "Team Chat",
                "full_category": "Communication > Team Chat",
                "confidence": 0.95,
                "reasoning": "stub",
                "backend": "test",
            },
        ]

        pipeline = ClassificationPipeline(
            aw_client=mock_aw,
            classifier=classifier,
            gemma_oracle=oracle,
            confidence_threshold=0.75,
        )

        stats = pipeline.run(
            source_bucket_ids=["source_bucket_1"],
            target_bucket_id="test-target",
            limit_per_bucket=10,
        )

        assert stats["total_events_processed"] == 2
        assert stats["events_saved_to_new_bucket"] == 2
        assert stats["gemma_oracle_calls"] == 2  # Initially untrained, so Gemma handles them
        assert classifier.is_trained
        assert len(classifier.training_samples) == 2

        # Verify that original buckets were NOT modified in place and mock_aw.insert_events was called for target bucket
        mock_aw.insert_events.assert_called_once()
        args, kwargs = mock_aw.insert_events.call_args
        target_bid, inserted_events = args
        assert target_bid == "test-target"
        assert inserted_events[0]["data"]["category"]
        assert "subcategory" in inserted_events[0]["data"]
        assert "full_category" in inserted_events[0]["data"]
        assert "category_hierarchy" in inserted_events[0]["data"]
        assert "classified_by" in inserted_events[0]["data"]
        assert "confidence" in inserted_events[0]["data"]


def test_pipeline_with_afk_filtering():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.joblib"
        data_path = Path(tmpdir) / "data.jsonl"

        mock_aw = MagicMock()
        mock_aw.find_afk_bucket_for_source.return_value = None
        mock_aw.create_bucket.return_value = {"id": "test-target", "created": True}
        mock_aw.find_afk_bucket_for_source.return_value = "aw-watcher-afk_fedora"

        def mock_get_events(bucket_id, **kwargs):
            if "afk" in bucket_id:
                return [
                    # User was not-afk only from 12:00:00 to 12:00:45
                    {
                        "timestamp": "2026-11-05T12:00:00Z",
                        "duration": 45.0,
                        "data": {"status": "not-afk"},
                    },
                    # User was AFK from 12:00:45 onwards
                    {
                        "timestamp": "2026-11-05T12:00:45Z",
                        "duration": 1800.0,
                        "data": {"status": "afk"},
                    },
                ]
            else:
                return [
                    # Event 1: starts at 12:00:00, duration 30s -> NOT AFK (kept)
                    {
                        "timestamp": "2026-11-05T12:00:00Z",
                        "duration": 30.0,
                        "data": {"app": "Terminal", "title": "git status"},
                    },
                    # Event 2: starts at 12:05:00, duration 120s -> FULLY AFK (must be dropped)
                    {
                        "timestamp": "2026-11-05T12:05:00Z",
                        "duration": 120.0,
                        "data": {"app": "Firefox", "title": "Idle Video"},
                    },
                ]

        mock_aw.get_events.side_effect = mock_get_events
        mock_aw.insert_events.side_effect = lambda bucket_id, evs: evs

        classifier = FastClassifier(model_path=model_path, training_data_path=data_path)
        oracle = GemmaOracle(backend="heuristic")

        pipeline = ClassificationPipeline(
            aw_client=mock_aw,
            classifier=classifier,
            gemma_oracle=oracle,
            confidence_threshold=0.75,
        )

        stats = pipeline.run(
            source_bucket_ids=["aw-watcher-window_fedora"],
            target_bucket_id="test-target",
            limit_per_bucket=10,
            filter_afk=True,
        )

        assert stats["afk_filtering_enabled"] is True
        assert "aw-watcher-afk_fedora" in stats["afk_buckets_used"]
        assert stats["total_raw_events_fetched"] == 2
        assert stats["afk_events_filtered_out"] == 1
        assert stats["total_events_processed"] == 1  # Only the active one was processed
        assert stats["events_saved_to_new_bucket"] == 1


def test_pipeline_with_time_range():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.joblib"
        data_path = Path(tmpdir) / "data.jsonl"

        mock_aw = MagicMock()
        mock_aw.create_bucket.return_value = {"id": "test-target", "created": True}
        mock_aw.get_events.return_value = [
            {
                "timestamp": "2026-11-06T08:00:00Z",
                "duration": 60.0,
                "data": {"app": "Code", "title": "main.py"},
            }
        ]
        mock_aw.insert_events.side_effect = lambda bucket_id, evs: evs

        classifier = FastClassifier(model_path=model_path, training_data_path=data_path)
        oracle = GemmaOracle(backend="heuristic")
        pipeline = ClassificationPipeline(aw_client=mock_aw, classifier=classifier, gemma_oracle=oracle)

        stats = pipeline.run(
            source_bucket_ids=["aw-watcher-window_laptop"],
            target_bucket_id="test-target",
            time_range="24h",
            filter_afk=False,
        )

        assert stats["time_range"] == "24h"
        assert stats["start_time"] is not None
        assert stats["end_time"] is not None
        assert stats["total_events_processed"] == 1

        # Check mock_aw.get_events was called with start and end
        args, kwargs = mock_aw.get_events.call_args
        assert kwargs.get("start") is not None
        assert kwargs.get("end") is not None


def test_pipeline_no_fast_classifier_training_samples():
    """Verify that training samples are ONLY added by Gemma/user, never by fast_classifier."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.joblib"
        data_path = Path(tmpdir) / "data.jsonl"

        mock_aw = MagicMock()
        mock_aw.create_bucket.return_value = {"id": "test-target", "created": True}
        mock_aw.find_afk_bucket_for_source.return_value = None
        mock_aw.get_events.return_value = [
            {"timestamp": "2026-11-06T08:00:00Z", "duration": 60.0, "data": {"app": "firefox", "title": "Google Search", "url": "https://google.com/search?q=test"}},
        ]
        mock_aw.insert_events.side_effect = lambda bucket_id, evs: evs

        classifier = FastClassifier(model_path=model_path, training_data_path=data_path)
        # Pre-seed classifier with at least 2 classes so logistic regression can fit
        classifier.add_training_sample("app:foot domain: root_domain: path: title:<user>@laptop:~ url:", "Development", "DevOps & Terminal", source="gemma_e2b")
        classifier.add_training_sample("app:firefox domain:google.com root_domain:google.com path:search title:google search url:https://google.com", "Browsing & Research", "Search & Exploration", source="gemma_e2b")
        retrained = classifier.retrain_full()
        assert retrained is True
        assert classifier.is_trained is True
        initial_samples_count = len(classifier.training_samples)

        oracle = GemmaOracle(backend="heuristic")
        pipeline = ClassificationPipeline(aw_client=mock_aw, classifier=classifier, gemma_oracle=oracle, confidence_threshold=0.3)

        stats = pipeline.run(
            source_bucket_ids=["aw-watcher-window_laptop"],
            target_bucket_id="test-target",
            collect_training_data=True,
            filter_afk=False,
        )

        assert stats["fast_classifier_calls"] == 1
        assert stats["gemma_oracle_calls"] == 0
        # No new samples added by fast_classifier
        assert len(classifier.training_samples) == initial_samples_count
        assert not any(s.get("source") == "fast_classifier" for s in classifier.training_samples)


def test_pipeline_device_afk_isolation():
    """Verify that phone activities are not filtered by laptop AFK status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.joblib"
        data_path = Path(tmpdir) / "data.jsonl"

        mock_aw = MagicMock()
        mock_aw.create_bucket.return_value = {"id": "test-target", "created": True}
        mock_aw.list_buckets.return_value = {
            "aw-watcher-android-synced-from-oneplus_12": {"hostname": "oneplus_12", "type": "currentwindow"},
            "aw-watcher-afk_laptop": {"hostname": "laptop", "type": "afkstatus"},
        }
        # Phone has no AFK bucket
        mock_aw.find_afk_bucket_for_source.side_effect = lambda bid: None if "oneplus" in bid else "aw-watcher-afk_laptop"
        mock_aw.get_events.return_value = [
            {"timestamp": "2026-11-06T08:00:00Z", "duration": 60.0, "data": {"app": "com.android.chrome", "title": "Mobile News"}},
        ]
        mock_aw.insert_events.side_effect = lambda bucket_id, evs: evs

        classifier = FastClassifier(model_path=model_path, training_data_path=data_path)
        oracle = GemmaOracle(backend="heuristic")
        pipeline = ClassificationPipeline(aw_client=mock_aw, classifier=classifier, gemma_oracle=oracle)

        stats = pipeline.run(
            source_bucket_ids=["aw-watcher-android-synced-from-oneplus_12"],
            target_bucket_id="test-target",
            filter_afk=True,
        )

        assert stats["afk_events_filtered_out"] == 0
        assert stats["total_events_processed"] == 1
        assert "aw-watcher-afk_laptop" not in stats["afk_buckets_used"]


def test_pipeline_audible_tab_afk_exception():
    """Verify that tabs playing video/movies exempt user from AFK during pipeline run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.joblib"
        data_path = Path(tmpdir) / "data.jsonl"

        mock_aw = MagicMock()
        mock_aw.create_bucket.return_value = {"id": "test-target", "created": True}
        mock_aw.find_afk_bucket_for_source.return_value = "aw-watcher-afk_fedora"
        mock_aw.find_web_buckets_for_source.return_value = ["aw-watcher-web-firefox_fedora"]
        mock_aw.list_buckets.return_value = {
            "aw-watcher-window_fedora": {"hostname": "fedora", "type": "currentwindow"},
            "aw-watcher-afk_fedora": {"hostname": "fedora", "type": "afkstatus"},
            "aw-watcher-web-firefox_fedora": {"hostname": "fedora", "type": "web.tab.current"},
        }

        def mock_get_events(bucket_id, **kwargs):
            if "afk" in bucket_id:
                # User had zero mouse/keyboard movement (100% AFK)
                return [
                    {
                        "timestamp": "2026-11-05T12:00:00Z",
                        "duration": 1800.0,
                        "data": {"status": "afk"},
                    }
                ]
            elif "web" in bucket_id:
                # Web watcher reports YouTube video playing with audio for the first 20 minutes
                return [
                    {
                        "timestamp": "2026-11-05T12:00:00Z",
                        "duration": 1200.0,
                        "data": {
                            "audible": True,
                            "url": "https://www.youtube.com/watch?v=doc123",
                            "title": "Documentary - YouTube",
                        },
                    }
                ]
            else:
                # Window watcher events
                return [
                    # Event 1: watching documentary (1200s) -> should be kept due to tab audio
                    {
                        "timestamp": "2026-11-05T12:00:00Z",
                        "duration": 1200.0,
                        "data": {"app": "Firefox", "title": "Documentary - YouTube"},
                    },
                    # Event 2: idle tab after video ended (600s) -> should be dropped as AFK
                    {
                        "timestamp": "2026-11-05T12:20:00Z",
                        "duration": 600.0,
                        "data": {"app": "Firefox", "title": "Blank Tab"},
                    },
                ]

        mock_aw.get_events.side_effect = mock_get_events
        mock_aw.insert_events.side_effect = lambda bucket_id, evs: evs

        classifier = FastClassifier(model_path=model_path, training_data_path=data_path)
        oracle = GemmaOracle(backend="heuristic")
        pipeline = ClassificationPipeline(aw_client=mock_aw, classifier=classifier, gemma_oracle=oracle)

        stats = pipeline.run(
            source_bucket_ids=["aw-watcher-window_fedora"],
            target_bucket_id="test-target",
            filter_afk=True,
            include_audible=True,
        )

        assert stats["afk_filtering_enabled"] is True
        assert stats["audible_filtering_enabled"] is True
        assert stats["total_raw_events_fetched"] == 2
        assert stats["afk_events_filtered_out"] == 1
        assert stats["total_events_processed"] == 1
        assert stats["events_saved_to_new_bucket"] == 1
        assert "aw-watcher-afk_fedora" in stats["afk_buckets_used"]
        assert "aw-watcher-web-firefox_fedora" in stats["audible_buckets_used"]


def test_pipeline_skips_zero_duration_events():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.joblib"
        data_path = Path(tmpdir) / "data.jsonl"

        mock_aw = MagicMock()
        mock_aw.find_afk_bucket_for_source.return_value = None
        mock_aw.create_bucket.return_value = {"id": "test-target", "created": True}
        mock_aw.get_events.return_value = [
            {"timestamp": "2026-11-05T12:00:00Z", "duration": 45.0, "data": {"app": "foot", "title": "terminal"}},
            {"timestamp": "2026-11-05T12:01:00Z", "duration": 0.0, "data": {"app": "foot", "title": "flicker 0s"}},
            {"timestamp": "2026-11-05T12:02:00Z", "duration": -5.0, "data": {"app": "foot", "title": "negative 0s"}},
            {"timestamp": "2026-11-05T12:03:00Z", "duration": 30.0, "data": {"app": "Brave-browser", "title": "Gmail", "url": "https://mail.google.com"}},
        ]
        mock_aw.insert_events.side_effect = lambda bucket_id, evs: evs

        classifier = FastClassifier(model_path=model_path, training_data_path=data_path)
        oracle = GemmaOracle(backend="heuristic")
        pipeline = ClassificationPipeline(aw_client=mock_aw, classifier=classifier, gemma_oracle=oracle)

        stats = pipeline.run(
            source_bucket_ids=["source_bucket_1"],
            target_bucket_id="test-target",
            filter_afk=False,
        )

        # 0s and negative events must be skipped
        assert stats["total_raw_events_fetched"] == 2
        assert stats["total_events_processed"] == 2
        # And the Gmail browser activity must be Communication > Email, not Browsing
        cat_dist = stats["category_distribution"]
        assert cat_dist.get("Communication") == 1
        assert cat_dist.get("Browsing & Research", 0) == 0


def test_pipeline_target_deduplication_and_force_reclassify():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.joblib"
        data_path = Path(tmpdir) / "data.jsonl"

        mock_aw = MagicMock()
        mock_aw.find_afk_bucket_for_source.return_value = None
        mock_aw.create_bucket.return_value = {"id": "test-target", "created": True}
        mock_aw.get_events.return_value = [
            {"timestamp": "2026-11-05T12:00:00Z", "duration": 60.0, "data": {"app": "foot", "title": "term"}},
            {"timestamp": "2026-11-05T12:01:00Z", "duration": 60.0, "data": {"app": "slack", "title": "chat"}},
        ]
        # Target already has the first event (12:00:00Z)
        mock_aw.get_existing_target_timestamps.return_value = {"2026-11-05T12:00:00Z"}
        mock_aw.insert_events.side_effect = lambda bucket_id, evs: evs

        classifier = FastClassifier(model_path=model_path, training_data_path=data_path)
        oracle = GemmaOracle(backend="heuristic")
        pipeline = ClassificationPipeline(aw_client=mock_aw, classifier=classifier, gemma_oracle=oracle)

        # 1. Normal run: deduplication skips the first event
        stats = pipeline.run(
            source_bucket_ids=["source_bucket_1"],
            target_bucket_id="test-target",
            filter_afk=False,
            force_reclassify=False,
        )
        assert stats["already_categorized_events_skipped"] == 1
        assert stats["total_events_processed"] == 1
        assert stats["events_saved_to_new_bucket"] == 1

        # 2. Force run: reclassifies both events
        stats_force = pipeline.run(
            source_bucket_ids=["source_bucket_1"],
            target_bucket_id="test-target",
            filter_afk=False,
            force_reclassify=True,
        )
        assert stats_force["already_categorized_events_skipped"] == 0
        assert stats_force["total_events_processed"] == 2
        assert stats_force["events_saved_to_new_bucket"] == 2





def test_pipeline_does_not_learn_from_heuristic_fallback():
    """An LLM outage falls back to the heuristic; those hard-coded labels must not be cached."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_aw = MagicMock()
        mock_aw.find_afk_bucket_for_source.return_value = None
        mock_aw.get_events.return_value = [
            {"timestamp": "2026-11-05T12:00:00Z", "duration": 60.0,
             "data": {"app": "customtool", "title": "Quarterly planning board"}},
        ]
        mock_aw.insert_events.side_effect = lambda bucket_id, evs: evs

        classifier = FastClassifier(
            model_path=Path(tmpdir) / "model.joblib", training_data_path=Path(tmpdir) / "data.jsonl"
        )
        # Unreachable server -> every activity falls back to the heuristic
        oracle = GemmaOracle(backend="openai_compatible", openai_url="http://127.0.0.1:9/v1")

        stats = ClassificationPipeline(
            aw_client=mock_aw, classifier=classifier, gemma_oracle=oracle, confidence_threshold=0.75
        ).run(source_bucket_ids=["source_bucket_1"], target_bucket_id="t", limit_per_bucket=10)

        assert stats["gemma_oracle_calls"] == 1
        assert stats["new_training_samples_added"] == 0
        assert classifier.training_samples == []
