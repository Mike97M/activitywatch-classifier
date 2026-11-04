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
                "data": {"app": "foot", "title": "jdoe@laptop:~"},
            },
            {
                "timestamp": "2026-11-05T12:01:00Z",
                "duration": 120.0,
                "data": {"app": "Brave-browser", "title": "Home / X", "url": "https://x.com/home"},
            },
        ]
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
        assert inserted_events[0]["data"]["category"] in ["Development", "Social Media"]
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

