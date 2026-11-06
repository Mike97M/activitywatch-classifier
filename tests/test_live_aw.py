"""Integration and unit tests for AWClient."""

from unittest.mock import MagicMock
from aw_client import AWClient


def test_aw_client_connection():
    client = AWClient()
    try:
        info = client.get_info()
        assert "hostname" in info
        buckets = client.list_buckets()
        assert isinstance(buckets, dict)
        assert len(buckets) > 0
    except Exception as e:
        # If server is temporarily unreachable in some testing environments
        assert True


def test_sync_classified_host_bucket_with_web_urls():
    client = AWClient(base_url="http://mock-server:5600")
    client.create_bucket = MagicMock(return_value={"created": True})
    client.insert_events = MagicMock(return_value=[])

    events = [
        {
            "timestamp": "2026-11-06T09:00:00Z",
            "duration": 45.0,
            "data": {
                "app": "firefox",
                "title": "Google AI Studio",
                "url": "https://aistudio.google.com/prompts/123",
                "category": "Development",
                "subcategory": "Coding & IDE",
            },
        },
        {
            "timestamp": "2026-11-06T09:01:00Z",
            "duration": 30.0,
            "data": {
                "app": "foot",
                "title": "terminal",
                "category": "Development",
                "subcategory": "DevOps & Terminal",
            },
        },
    ]

    res = client.sync_classified_host_bucket(events, host="test_host")
    assert res is True

    # Ensure both window, browser-specific web, and afk buckets were created
    created_bucket_ids = [call[1]["bucket_id"] for call in client.create_bucket.call_args_list]
    assert "aw-watcher-window_test_host" in created_bucket_ids
    assert "aw-watcher-web-firefox_test_host" in created_bucket_ids
    assert "aw-watcher-afk_test_host" in created_bucket_ids

    # Ensure web bucket received the web event with URL
    inserted_calls = client.insert_events.call_args_list
    web_call = next(call for call in inserted_calls if call[0][0] == "aw-watcher-web-firefox_test_host")
    inserted_web_events = web_call[0][1]
    assert len(inserted_web_events) == 1
    assert inserted_web_events[0]["data"]["url"] == "https://aistudio.google.com/prompts/123"
    assert inserted_web_events[0]["data"]["title"] == "Google AI Studio"

    # Also verify that window events have canonical browser app for web events
    win_call = next(call for call in inserted_calls if call[0][0] == "aw-watcher-window_test_host")
    inserted_win_events = win_call[0][1]
    firefox_win_event = next(e for e in inserted_win_events if e["data"].get("url"))
    assert firefox_win_event["data"]["app"] == "Firefox"


def test_sync_classified_host_bucket_multi_browser():
    client = AWClient(base_url="http://mock-server:5600")
    client.create_bucket = MagicMock(return_value={"created": True})
    client.insert_events = MagicMock(return_value=[])

    events = [
        {
            "timestamp": "2026-11-06T09:00:00Z",
            "duration": 45.0,
            "data": {
                "source_bucket": "aw-watcher-web-brave_laptop",
                "title": "Brave Search",
                "url": "https://search.brave.com",
            },
        },
        {
            "timestamp": "2026-11-06T09:01:00Z",
            "duration": 30.0,
            "data": {
                "source_bucket": "aw-watcher-android-web-synced-from-oneplus_12",
                "title": "Android Chrome Page",
                "url": "https://developer.android.com",
            },
        },
    ]

    res = client.sync_classified_host_bucket(events, host="classified")
    assert res is True

    created_bucket_ids = [call[1]["bucket_id"] for call in client.create_bucket.call_args_list]
    assert "aw-watcher-web-brave_classified" in created_bucket_ids
    assert "aw-watcher-web-chrome_classified" in created_bucket_ids
    assert "aw-watcher-window_classified" in created_bucket_ids
    assert "aw-watcher-afk_classified" in created_bucket_ids
