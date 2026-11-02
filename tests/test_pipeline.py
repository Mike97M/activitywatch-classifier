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
        assert "classified_by" in inserted_events[0]["data"]
        assert "confidence" in inserted_events[0]["data"]
