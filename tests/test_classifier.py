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
        cat, conf, probs = clf.predict("app:unknownapp title:something new")
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

        test_feat = "app:myeditor title:refactor/auth.py"
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


def _temp_classifier(tmpdir) -> FastClassifier:
    return FastClassifier(
        model_path=Path(tmpdir) / "model.joblib",
        training_data_path=Path(tmpdir) / "data.jsonl",
    )


def test_rules_cascade_high_precision_cases():
    """Deterministic rules must handle obvious apps/domains without the LLM."""
    with tempfile.TemporaryDirectory() as tmpdir:
        clf = _temp_classifier(tmpdir)

        cases = {
            "app:brave domain:github.com root_domain:github.com title:pull request #1 url:https://github.com/a/b": (
                "Development", "Git & Code Review"
            ),
            "app:brave domain:docs.google.com root_domain:google.com title:budget url:https://docs.google.com/x": (
                "Productivity & Planning", "Docs & Spreadsheets"
            ),
            "app:foot domain: root_domain: title:btop - system resource monitor url:": (
                "System & Admin", "Hardware & System Monitor"
            ),
            "app:foot domain: root_domain: title:sudo zypper dup url:": (
                "System & Admin", "Package Management"
            ),
            "app:duolingo domain: root_domain: title:Duolingo: Language Practice url:": (
                "Education & Learning", "Language Learning"
            ),
            "app:brave domain:mail.google.com root_domain:google.com title:inbox url:https://mail.google.com/": (
                "Communication", "Email"
            ),
        }
        for feature_text, expected in cases.items():
            cat, sub, full, conf, _ = clf.predict_hierarchy(feature_text)
            assert (cat, sub) == expected, f"{feature_text} -> {(cat, sub)}, expected {expected}"
            assert conf >= 0.9


def test_cache_key_folds_volatile_numbers_and_query_strings():
    with tempfile.TemporaryDirectory() as tmpdir:
        clf = _temp_classifier(tmpdir)
        clf.add_training_sample(
            "app:myeditor title:build #42 url:https://example.org/run/42?token=x",
            "Development",
            "Debugging & Testing",
            retrain_now=False,
        )
        # Same activity with different run number and tracking query string -> cache hit
        cat, sub, full, conf, _ = clf.predict_hierarchy(
            "app:myeditor title:build #99 url:https://example.org/run/99?ref=feed"
        )
        assert (cat, sub) == ("Development", "Debugging & Testing")
        assert conf == 1.0


def test_human_correction_is_never_overwritten_by_teacher():
    with tempfile.TemporaryDirectory() as tmpdir:
        clf = _temp_classifier(tmpdir)
        feat = "app:myeditor title:weekly report"
        clf.add_training_sample(feat, "Productivity & Planning", "Docs & Spreadsheets", source="gemma_e2b", retrain_now=False)

        # Human fixes the label.
        assert clf.add_training_sample(feat, "Productivity & Planning", "Notes & Knowledge Base", source="human_correction", retrain_now=False)
        cat, sub, full, conf, _ = clf.predict_hierarchy(feat)
        assert (cat, sub) == ("Productivity & Planning", "Notes & Knowledge Base")

        # Teacher tries to overwrite -> rejected.
        assert not clf.add_training_sample(feat, "Development", "Coding & IDE", source="gemma_e2b", retrain_now=False)
        cat, sub, full, conf, _ = clf.predict_hierarchy(feat)
        assert (cat, sub) == ("Productivity & Planning", "Notes & Knowledge Base")
        assert conf == 1.0


def test_predict_batch_matches_predict_hierarchy():
    with tempfile.TemporaryDirectory() as tmpdir:
        clf = _temp_classifier(tmpdir)
        feats = [
            "app:brave domain:x.com root_domain:x.com title:home / x url:https://x.com/home",
            "app:unknownapp title:something new",
        ]
        batch = clf.predict_batch(feats)
        for feat, (full, conf, probs) in zip(feats, batch):
            cat, sub, full2, conf2, probs2 = clf.predict_hierarchy(feat)
            assert (full, conf) == (full2, conf2)


def test_concurrent_corrections_and_predictions_do_not_corrupt_state(tmp_path):
    import threading

    clf = FastClassifier(model_path=tmp_path / "m.joblib", training_data_path=tmp_path / "d.jsonl")
    errors = []

    def add(n):
        try:
            for i in range(15):
                clf.add_training_sample(f"app:tool{n}x{i} title:t{i}", "Development" if i % 2 else "Communication",
                                        "Coding & IDE" if i % 2 else "Email", source="human_correction")
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def predict():
        try:
            for _ in range(30):
                clf.predict_batch_meta(["app:tool0x1 title:t1", "app:other title:x"])
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=add, args=(n,)) for n in range(3)] + [threading.Thread(target=predict)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(clf.training_samples) == 45
    assert len((tmp_path / "d.jsonl").read_text().splitlines()) == 45


def test_rules_fixed_by_benchmark_v2():
    def rule(app, title, url=""):
        r = FastClassifier.match_rules(FastClassifier.extract_features({"app": app, "title": title, "url": url}))
        return " > ".join(r) if r else None

    assert rule("teams", "Meeting with Design | Microsoft Teams") == "Communication > Calls & Meetings"
    assert rule("teams", "Chat | Backend Squad | Microsoft Teams") == "Communication > Team Chat"
    assert rule("calibre", "E-book viewer - The Pragmatic Programmer") == "Education & Learning > Reading & Reference"
    assert rule("firefox", "Kindle Cloud Reader", "https://read.amazon.com/") == "Education & Learning > Reading & Reference"
    assert rule("firefox", "Amazon.com : keyboards", "https://www.amazon.com/s?k=x") == "Browsing & Research > Shopping & E-Commerce"
    assert rule("firefox", "Show HN | Hacker News", "https://news.ycombinator.com/item?id=1") == (
        "Browsing & Research > Technical Articles & Papers"
    )


def test_rules_fixed_by_real_data_review():
    def rule(app, title, url=""):
        r = FastClassifier.match_rules(FastClassifier.extract_features({"app": app, "title": title, "url": url}))
        return " > ".join(r) if r else None

    # a project folder named after ActivityWatch is not the ActivityWatch monitor
    assert rule("foot", "activitywatch-classifier: limactl - shell") == "Development > DevOps & Terminal"
    assert rule("foot", "btop") == "System & Admin > Hardware & System Monitor"
    assert rule("foot", "sudo zypper dup") == "System & Admin > Package Management"
    # cloud consoles are infrastructure, not shopping / search
    assert rule("Brave-browser", "Create user | IAM", "https://us-east-1.console.aws.amazon.com/iam/home") == (
        "Development > DevOps & Terminal")
    assert rule("Brave-browser", "Credentials", "https://console.cloud.google.com/apis/credentials") == (
        "Development > DevOps & Terminal")
    assert rule("Brave-browser", "Amazon.de", "https://www.amazon.de/") == "Browsing & Research > Shopping & E-Commerce"
    # job search
    jobs = "Career & Jobs > Job Search & Applications"
    assert rule("Brave-browser", "Apply", "https://jobs.lever.co/acme/123/apply") == jobs
    assert rule("Brave-browser", "Careers", "https://www.acme.com/careers/data-engineer") == jobs
    assert rule("Brave-browser", "Jobs | LinkedIn", "https://www.linkedin.com/jobs/search") == jobs
    assert rule("Brave-browser", "Feed | LinkedIn", "https://www.linkedin.com/feed/") == "Social Media > LinkedIn & Professional"
    assert rule("Brave-browser", "Build", "https://github.com/acme/api/actions/runs/1/job/2") != jobs
