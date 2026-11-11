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
    # Verify distinct subcategory color
    assert devops_class["data"]["color"] == "#06b6d4"
    root_dev = next(c for c in classes if c["name"] == ["Development"])
    assert root_dev["data"]["color"] == "#4DBA87"


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


def test_education_category_and_duolingo():
    classes = build_activitywatch_classes()
    edu_class = next((c for c in classes if c["name"] == ["Education & Learning", "Language Learning"]), None)
    assert edu_class is not None
    assert "duolingo" in edu_class["rule"]["regex"]

    # Verify Duolingo is NOT in gaming
    gaming_class = next(c for c in classes if c["name"] == ["Media & Entertainment", "Gaming"])
    assert "duolingo" not in gaming_class["rule"]["regex"]


def test_browser_apps_not_bound_to_browsing():
    """Verify generic browsers are not locked into Browsing so they dynamically reflect active tabs."""
    classes = build_activitywatch_classes()
    root_browsing = next(c for c in classes if c["name"] == ["Browsing & Research"])
    root_regex = root_browsing["rule"].get("regex", "")
    assert "firefox" not in root_regex
    assert "chrome" not in root_regex
    assert "brave-browser" not in root_regex
    assert "|brave|" not in f"|{root_regex}|"

    # Subcategory classes (depth 2) have specific search/article rules
    sub_search = next(c for c in classes if c["name"] == ["Browsing & Research", "Search & Exploration"])
    assert "firefox" not in sub_search["rule"]["regex"]
    assert "duckduckgo" in sub_search["rule"]["regex"]


def test_subcategory_distinct_colors():
    """Verify subcategories receive distinct, identifiable color shades."""
    classes = build_activitywatch_classes()
    coding = next(c for c in classes if c["name"] == ["Development", "Coding & IDE"])
    devops = next(c for c in classes if c["name"] == ["Development", "DevOps & Terminal"])
    git = next(c for c in classes if c["name"] == ["Development", "Git & Code Review"])

    # Colors must be different across subcategories
    assert coding["data"]["color"] != devops["data"]["color"]
    assert devops["data"]["color"] != git["data"]["color"]
    assert coding["data"]["color"] == "#10b981"
    assert devops["data"]["color"] == "#06b6d4"
    assert git["data"]["color"] == "#34d399"


def test_finance_taxonomy_structure():
    from config import DEFAULT_TAXONOMY, flatten_taxonomy
    paths = flatten_taxonomy(DEFAULT_TAXONOMY)
    finance_paths = [p for p in paths if p.startswith("Finance & Banking")]
    assert "Finance & Banking > Banking & Payments" in finance_paths
    assert "Finance & Banking > Investments & Trading" in finance_paths


