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


def test_classifier_extract_features_domain_and_url():
    event_data = {
        "app": "firefox",
        "title": "Install Awacther On openSUSE | Google AI Studio",
        "url": "https://aistudio.google.com/prompts/1r4qFlm2qdyyZqOB7ezsAmmYuSvvFswoz",
    }
    feat = FastClassifier.extract_features(event_data)
    assert "domain:aistudio.google.com" in feat
    assert "root_domain:google.com" in feat
    assert "path:prompts" in feat
    assert "app:firefox" in feat


def test_browser_email_classification_not_browsing():
    clf = FastClassifier()
    browser_emails = [
        {"app": "Brave-browser", "title": "Inbox (5) - Gmail - Brave", "url": "https://mail.google.com/mail/u/0/#inbox"},
        {"app": "Brave-browser", "title": "Inbox - app.mailbox.org - Brave", "url": "https://app.mailbox.org/appsuite/#app=io.ox/mail"},
        {"app": "Brave-browser", "title": "Mail - John Doe - Outlook", "url": "https://outlook.live.com/mail/0/inbox"},
        {"app": "Brave-browser", "title": "Proton Mail", "url": "https://mail.proton.me/u/0/inbox"},
        {"app": "Brave-browser", "title": "Yahoo Mail", "url": "https://mail.yahoo.com/d/folders/1"},
        {"app": "Brave-browser", "title": "Fastmail", "url": "https://app.fastmail.com/mail/Inbox"},
        {"app": "Brave-browser", "title": "Webmail Login", "url": "https://webmail.example.com/"},
        {"app": "Brave-browser", "title": "Inbox (3) - Gmail - Brave", "url": ""},
        {"app": "Brave-browser", "title": "app.mailbox.org/appsuite/#app=io.ox/mail - Brave", "url": ""},
        {"app": "Brave-browser", "title": "Proton Mail - Brave", "url": ""},
        {"app": "Brave-browser", "title": "Mail - Outlook - Brave", "url": ""},
        {"app": "Firefox", "title": "Gmail", "url": "https://mail.google.com/mail/mu/mp/466/"},
        {"app": "Firefox", "title": "app.mailbox.org", "url": "https://app.mailbox.org/"},
    ]

    for ev in browser_emails:
        feat = clf.extract_features(ev)
        cat, subcat, full, conf, _ = clf.predict_hierarchy(feat)
        assert cat == "Communication", f"Expected Communication for {ev}, got {cat}"
        assert subcat == "Email", f"Expected Email for {ev}, got {subcat}"
        assert cat != "Browsing & Research", f"Should never be Browsing & Research: {ev}"


def test_classifier_batch_retraining_trigger():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.joblib"
        data_path = Path(tmpdir) / "data.jsonl"
        clf = FastClassifier(
            model_path=model_path,
            training_data_path=data_path,
        )

        # Add initial 2 samples with retrain_now=True to bootstrap
        clf.add_training_sample("app:foot title:terminal", "Development", "DevOps & Terminal", retrain_now=True)
        clf.add_training_sample("app:slack title:team", "Communication", "Team Chat", retrain_now=True)
        assert clf.is_trained
        assert clf._unretrained_count == 0

        # Add 9 samples with retrain_now=False, retrain_batch_size=10
        for i in range(9):
            clf.add_training_sample(
                f"app:browser title:doc_{i}",
                "Browsing & Research",
                "Search & Exploration",
                retrain_now=False,
                retrain_batch_size=10,
            )
        assert clf._unretrained_count == 9

        # Add 10th sample -> should trigger automatic retraining
        clf.add_training_sample(
            "app:browser title:doc_final",
            "Browsing & Research",
            "Search & Exploration",
            retrain_now=False,
            retrain_batch_size=10,
        )
        assert clf._unretrained_count == 0
        assert clf.get_stats()["total_training_samples"] == 12


def test_classifier_exact_memory_cache():
    """Verify that previously categorized activities are retrieved from memory cache with 1.0 confidence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.joblib"
        data_path = Path(tmpdir) / "data.jsonl"
        clf = FastClassifier(
            model_path=model_path,
            training_data_path=data_path,
        )

        test_feat = "app:code domain:github.com title:refactor/auth.py"
        clf.add_training_sample(
            feature_text=test_feat,
            category="Development",
            subcategory="Coding & IDE",
            retrain_now=False,
        )

        assert test_feat in clf.samples_by_text
        assert clf.get_stats()["cached_samples_count"] == 1

        # Predict hierarchy directly: even if untrained on disk, memory cache returns 1.0 confidence
        cat, subcat, full, conf, probs = clf.predict_hierarchy(test_feat)
        assert cat == "Development"
        assert subcat == "Coding & IDE"
        assert full == "Development > Coding & IDE"
        assert conf == 1.0
        assert probs[full] == 1.0
