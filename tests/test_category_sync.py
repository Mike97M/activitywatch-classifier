"""Unit tests for ActivityWatch category synchronization and Activity tab integration."""

from unittest.mock import MagicMock
from category_sync import build_activitywatch_classes, sync_taxonomy_to_activitywatch
from config import DEFAULT_TAXONOMY


def test_build_activitywatch_classes_structure():
    classes = build_activitywatch_classes()
    assert len(classes) >= 30

    # Verify hierarchical naming
    names = [c["name"] for c in classes]
    assert ["Development"] in names
    assert ["Development", "DevOps & Terminal"] in names
    assert ["Communication", "Team Chat"] in names
    assert ["Uncategorized"] in names

    # Verify rule structure
    devops_class = next(c for c in classes if c["name"] == ["Development", "DevOps & Terminal"])
    assert devops_class["rule"]["type"] == "regex"
    assert "foot" in devops_class["rule"]["regex"]
    assert devops_class["rule"]["ignore_case"] is True
    assert devops_class["data"]["color"] == "#4DBA87"


def test_build_activitywatch_classes_with_training_samples():
    training_samples = [
        {
            "text": "app:firefox domain:aistudio.google.com root_domain:google.com title:AI Studio",
            "category": "Development",
            "subcategory": "Coding & IDE",
        },
        {
            "text": "app:brave domain:app.slack.com root_domain:slack.com title:chat with team",
            "category": "Communication",
            "subcategory": "Team Chat",
        },
    ]

    classes = build_activitywatch_classes(training_samples=training_samples)
    coding_class = next(c for c in classes if c["name"] == ["Development", "Coding & IDE"])
    assert "aistudio\\.google\\.com" in coding_class["rule"]["regex"]

    chat_class = next(c for c in classes if c["name"] == ["Communication", "Team Chat"])
    assert "app\\.slack\\.com" in chat_class["rule"]["regex"]


def test_sync_taxonomy_to_activitywatch():
    mock_client = MagicMock()
    mock_client.set_setting.return_value = True

    mock_classifier = MagicMock()
    mock_classifier.taxonomy = DEFAULT_TAXONOMY
    mock_classifier.training_samples = []

    res = sync_taxonomy_to_activitywatch(mock_client, mock_classifier)
    assert res["success"] is True
    assert res["classes_count"] >= 30
    assert "Development" in res["categories"]


def test_generic_browsers_and_domains_not_polluting_categories():
    # Samples with browser app and generic root domain
    samples = [
        {
            "text": "app:brave-browser domain:mail.google.com root_domain:google.com title:gmail - brave",
            "category": "Communication",
            "subcategory": "Email",
        },
        {
            "text": "app:org.chromium.chromium domain:experiments.hetzner.com root_domain:hetzner.com title:hetzner",
            "category": "Browsing & Research",
            "subcategory": "Search & Exploration",
        },
    ]

    classes = build_activitywatch_classes(training_samples=samples)
    email_class = next(c for c in classes if c["name"] == ["Communication", "Email"])
    email_regex = email_class["rule"]["regex"]

    # Must include mail.google.com
    assert "mail\\.google\\.com" in email_regex
    # Must NOT include browser app names or generic root domain google.com (standalone)
    assert "brave-browser" not in email_regex
    assert "brave\\-browser" not in email_regex
    assert "(google\\.com)" not in email_regex
    assert "|google\\.com|" not in f"|{email_regex}|"
    assert "org.chromium.chromium" not in email_regex
