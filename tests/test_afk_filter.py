"""Unit tests for AFK filtering and interval intersection."""

from datetime import datetime, timezone
import pytest
from afk_filter import (
    filter_events_by_afk,
    filter_period_intersect,
    merge_intervals,
    parse_iso_timestamp,
    to_iso_timestamp,
)
from aw_client import AWClient


def test_parse_iso_timestamp_variations():
    # Nanosecond timestamp from ActivityWatch rust server
    ts1 = "2026-01-23T07:42:08.785888513Z"
    dt1 = parse_iso_timestamp(ts1)
    assert dt1.year == 2026
    assert dt1.month == 1
    assert dt1.day == 23
    assert dt1.hour == 7
    assert dt1.minute == 42
    assert dt1.tzinfo is not None

    # Timezone offset
    ts2 = "2026-01-23T09:42:08+02:00"
    dt2 = parse_iso_timestamp(ts2)
    assert dt2.hour == 9

    # Datetime object passed directly
    now = datetime.now(timezone.utc)
    assert parse_iso_timestamp(now) == now

    # Roundtrip conversion
    iso = to_iso_timestamp(dt1)
    assert iso.endswith("Z")


def test_merge_intervals():
    t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 1, 10, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 1, 10, 15, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 1, 1, 10, 25, 0, tzinfo=timezone.utc)
    t4 = datetime(2026, 1, 1, 10, 30, 0, tzinfo=timezone.utc)

    # Overlapping intervals: [10:00, 10:15] and [10:10, 10:25] -> [10:00, 10:25]
    merged = merge_intervals([(t0, t2), (t1, t3)])
    assert len(merged) == 1
    assert merged[0] == (t0, t3)

    # Disjoint intervals: [10:00, 10:10] and [10:25, 10:30]
    disjoint = merge_intervals([(t0, t1), (t3, t4)])
    assert len(disjoint) == 2
    assert disjoint[0] == (t0, t1)
    assert disjoint[1] == (t3, t4)


def test_filter_period_intersect_full_and_zero_overlap():
    events = [
        # Event 1: Fully within not-afk (10:05 to 10:15 = 600s)
        {
            "timestamp": "2026-01-01T10:05:00Z",
            "duration": 600.0,
            "data": {"app": "Neovim", "title": "main.py"},
        },
        # Event 2: Inside AFK period (10:35 to 10:45 = 600s)
        {
            "timestamp": "2026-01-01T10:35:00Z",
            "duration": 600.0,
            "data": {"app": "Firefox", "title": "YouTube (idle)"},
        },
    ]

    not_afk = [
        # Active from 10:00 to 10:30 (1800s)
        {"timestamp": "2026-01-01T10:00:00Z", "duration": 1800.0, "data": {"status": "not-afk"}}
    ]

    filtered = filter_period_intersect(events, not_afk)
    # Event 2 must be completely discarded
    assert len(filtered) == 1
    assert filtered[0]["data"]["app"] == "Neovim"
    assert filtered[0]["duration"] == 600.0
    assert filtered[0]["timestamp"] == "2026-01-01T10:05:00Z"


def test_filter_period_intersect_partial_overlap_trim():
    events = [
        # Event starts during not-afk at 10:25 and extends to 10:40 (user leaves at 10:30)
        {
            "timestamp": "2026-01-01T10:25:00Z",
            "duration": 900.0,  # 15 minutes
            "data": {"app": "VSCode", "title": "server.py"},
        }
    ]

    not_afk = [
        # User is active only until 10:30 (5 mins overlap with event)
        {"timestamp": "2026-01-01T10:00:00Z", "duration": 1800.0, "data": {"status": "not-afk"}}
    ]

    filtered = filter_period_intersect(events, not_afk)
    assert len(filtered) == 1
    # Active duration must be trimmed from 900s to 300s (5 mins)
    assert filtered[0]["duration"] == 300.0
    assert filtered[0]["timestamp"] == "2026-01-01T10:25:00Z"


def test_filter_period_intersect_point_events():
    events = [
        # Point event inside not-afk
        {"timestamp": "2026-01-01T10:15:00Z", "duration": 0.0, "data": {"app": "Terminal"}},
        # Point event inside afk
        {"timestamp": "2026-01-01T10:45:00Z", "duration": 0.0, "data": {"app": "Terminal"}},
    ]

    not_afk = [
        {"timestamp": "2026-01-01T10:00:00Z", "duration": 1800.0, "data": {"status": "not-afk"}}
    ]

    filtered = filter_period_intersect(events, not_afk)
    assert len(filtered) == 1
    assert filtered[0]["timestamp"] == "2026-01-01T10:15:00Z"


def test_filter_events_by_afk():
    raw_events = [
        {"timestamp": "2026-01-01T10:10:00Z", "duration": 120.0, "data": {"app": "Slack"}},
        {"timestamp": "2026-01-01T10:35:00Z", "duration": 300.0, "data": {"app": "Slack"}},
    ]

    # Mixed AFK stream
    afk_stream = [
        {"timestamp": "2026-01-01T10:00:00Z", "duration": 1200.0, "data": {"status": "not-afk"}},
        {"timestamp": "2026-01-01T10:20:00Z", "duration": 2400.0, "data": {"status": "afk"}},
    ]

    filtered = filter_events_by_afk(raw_events, afk_stream)
    assert len(filtered) == 1
    assert filtered[0]["data"]["app"] == "Slack"
    assert filtered[0]["timestamp"] == "2026-01-01T10:10:00Z"


def test_find_afk_bucket_for_source(monkeypatch):
    client = AWClient(base_url="http://mock-server:5600")

    mock_buckets = {
        "aw-watcher-window_fedora": {"hostname": "fedora", "type": "currentwindow"},
        "aw-watcher-afk_fedora": {"hostname": "fedora", "type": "afkstatus"},
        "aw-watcher-web-firefox_laptop": {"hostname": "laptop", "type": "web.tab.current"},
        "aw-watcher-afk_laptop": {"hostname": "laptop", "type": "afkstatus"},
    }

    monkeypatch.setattr(client, "list_buckets", lambda: mock_buckets)

    assert client.find_afk_bucket_for_source("aw-watcher-window_fedora") == "aw-watcher-afk_fedora"
    assert client.find_afk_bucket_for_source("aw-watcher-web-firefox_laptop") == "aw-watcher-afk_laptop"
