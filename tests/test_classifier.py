"""Unit tests for the FastClassifier."""

import os
import tempfile
from pathlib import Path
from classifier import FastClassifier


def test_classifier_untrained_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.joblib"
        data_path = Path(tmpdir) / "data.jsonl"
        clf = FastClassifier(
            categories=["Development", "Communication"],
            model_path=model_path,
            training_data_path=data_path,
        )

        assert not clf.is_trained
        cat, conf, probs = clf.predict("app:foot title:terminal")
        assert conf == 0.0
        assert "Uncategorized" in cat


def test_classifier_active_learning_flow():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.joblib"
        data_path = Path(tmpdir) / "data.jsonl"
        clf = FastClassifier(
            model_path=model_path,
            training_data_path=data_path,
        )

        # Add initial training samples with subcategories
        clf.add_training_sample("app:foot title:bash python script.py", "Development", "DevOps & Terminal")
        clf.add_training_sample("app:code title:pipeline.py - vscode", "Development", "Coding & IDE")
        clf.add_training_sample("app:slack title:#general team standup", "Communication", "Team Chat")
        clf.add_training_sample("app:telegram title:chat with alex", "Communication", "Team Chat")
        clf.add_training_sample("app:brave title:x.com / twitter feed", "Social Media", "X / Twitter")

        assert clf.is_trained
        assert clf.get_stats()["total_training_samples"] == 5

        # Predict new coding sample
        cat, subcat, full, conf, probs = clf.predict_hierarchy("app:foot domain:github.com title:git commit")
        assert cat == "Development"
        assert conf > 0.3
        assert 0.0 <= conf <= 1.0

        # Predict communication sample
        cat2, subcat2, full2, conf2, _ = clf.predict_hierarchy("app:slack title:product design channel")
        assert cat2 == "Communication"
        assert subcat2 == "Team Chat"
        assert conf2 > 0.3


def test_classifier_extract_features_anonymization():
    event_data = {
        "app": "foot",
        "title": "jdoe@laptop:/home/jdoe/Developement/secret",
        "url": "https://company.org/login?token=secret1234567890",
    }
    feat = FastClassifier.extract_features(event_data)
    assert "jdoe" not in feat
    assert "secret1234567890" not in feat
    assert "<user>" in feat or "<path>" in feat
    assert "redacted" in feat
