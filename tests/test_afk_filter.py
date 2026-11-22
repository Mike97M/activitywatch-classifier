"""Unit tests for AFK filtering and interval intersection."""

from datetime import datetime, timezone
import pytest
from afk_filter import (
    extract_audible_non_afk_intervals,
    filter_events_by_afk,
    filter_period_intersect,
    is_audible_non_afk,
    is_excluded_music,
    merge_intervals,
    parse_iso_timestamp,
    resolve_time_range,
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


def test_resolve_time_range():
    ref_time = datetime(2026, 11, 6, 15, 30, 0, tzinfo=timezone.utc)

    # Count / none
    assert resolve_time_range("count") == (None, None)
    assert resolve_time_range("all") == (None, None)

    # Today
    s_today, e_today = resolve_time_range("today", now=ref_time)
    assert s_today == "2026-11-06T00:00:00Z"
    assert e_today == "2026-11-06T15:30:00Z"

    # Today with local timezone (e.g. UTC+2): local midnight aligns with user's day
    from datetime import timedelta
    tz_plus2 = timezone(timedelta(hours=2))
    ref_time_tz2 = datetime(2026, 11, 6, 15, 30, 0, tzinfo=tz_plus2)
    s_tz2, e_tz2 = resolve_time_range("today", now=ref_time_tz2)
    assert s_tz2 == "2026-11-05T22:00:00Z"
    assert e_tz2 == "2026-11-06T13:30:00Z"

    # 24h / last_day
    s_24h, e_24h = resolve_time_range("24h", now=ref_time)
    assert s_24h == "2026-11-05T15:30:00Z"
    assert e_24h == "2026-11-06T15:30:00Z"

    # 7d / last_week
    s_7d, e_7d = resolve_time_range("7d", now=ref_time)
    assert s_7d == "2026-10-30T15:30:00Z"

    # 30d / last_month
    s_30d, e_30d = resolve_time_range("30d", now=ref_time)
    assert s_30d == "2026-10-07T15:30:00Z"

    # Custom start and end
    custom_s, custom_e = resolve_time_range(start_time="2026-11-01T08:00:00Z", end_time="2026-11-02T12:00:00Z")
    assert custom_s == "2026-11-01T08:00:00Z"
    assert custom_e == "2026-11-02T12:00:00Z"


def test_is_excluded_music():
    # Music services must be identified as excluded
    assert is_excluded_music({"url": "https://music.youtube.com/watch?v=123", "title": "Track"}) is True
    assert is_excluded_music({"url": "https://open.spotify.com/track/abc", "title": "Song"}) is True
    assert is_excluded_music({"url": "https://soundcloud.com/artist/track", "title": "Beat"}) is True
    assert is_excluded_music({"url": "https://deezer.com/track/456", "title": "Melody"}) is True
    assert is_excluded_music({"url": "https://tidal.com/browse/track/789", "title": "Audio"}) is True
    assert is_excluded_music({"url": "https://music.apple.com/us/album/xyz", "title": "Album"}) is True
    assert is_excluded_music({"title": "Favorite Song - YouTube Music"}) is True
    assert is_excluded_music({"title": "Artist Name - Spotify"}) is True
    assert is_excluded_music({"app": "spotify", "title": "Playing"}) is True

    # Movies, streams, and non-music videos must NOT be excluded
    assert is_excluded_music({"url": "https://www.youtube.com/watch?v=rpdTnPWFjDo", "title": "epstein files - YouTube"}) is False
    assert is_excluded_music({"url": "https://www.netflix.com/watch/80018294", "title": "Daredevil Season 1"}) is False
    assert is_excluded_music({"url": "https://www.twitch.tv/shroud", "title": "shroud playing Valorant"}) is False
    assert is_excluded_music({"url": "https://app.plex.tv/web/index.html", "title": "Inception (2010)"}) is False
    assert is_excluded_music({"url": "https://www.primevideo.com/detail/123", "title": "The Boys"}) is False


def test_is_audible_non_afk():
    # Video playing with audio
    assert is_audible_non_afk({
        "data": {"audible": True, "url": "https://www.youtube.com/watch?v=movie1", "title": "Documentary - YouTube"}
    }) is True

    # Video tab muted or paused (audible: False)
    assert is_audible_non_afk({
        "data": {"audible": False, "url": "https://www.youtube.com/watch?v=movie1", "title": "Documentary - YouTube"}
    }) is False

    # Tab playing music (audible: True, but YouTube Music) -> excluded
    assert is_audible_non_afk({
        "data": {"audible": True, "url": "https://music.youtube.com/watch?v=song1", "title": "Song - YouTube Music"}
    }) is False

    # Tab playing Spotify (audible: True) -> excluded
    assert is_audible_non_afk({
        "data": {"audible": True, "url": "https://open.spotify.com/track/123", "title": "Artist - Spotify"}
    }) is False


def test_extract_audible_non_afk_intervals():
    events = [
        # Normal heartbeat with duration 0.0 -> must be extended to pulsetime (65.0s)
        {
            "timestamp": "2026-11-05T12:00:00Z",
            "duration": 0.0,
            "data": {"audible": True, "url": "https://www.youtube.com/watch?v=1", "title": "Movie Part 1 - YouTube"},
        },
        # YouTube Music event -> must be ignored
        {
            "timestamp": "2026-11-05T12:01:00Z",
            "duration": 60.0,
            "data": {"audible": True, "url": "https://music.youtube.com/watch?v=2", "title": "Music Track"},
        },
        # Non-audible event -> must be ignored
        {
            "timestamp": "2026-11-05T12:02:00Z",
            "duration": 60.0,
            "data": {"audible": False, "url": "https://news.ycombinator.com", "title": "Hacker News"},
        },
    ]

    intervals = extract_audible_non_afk_intervals(events, pulsetime=65.0)
    assert len(intervals) == 1
    assert intervals[0]["timestamp"] == "2026-11-05T12:00:00Z"
    assert intervals[0]["duration"] == 65.0
    assert intervals[0]["data"]["status"] == "not-afk"
    assert intervals[0]["data"]["audible"] is True


def test_filter_events_by_afk_with_audible_movie():
    # User is AFK according to keyboard/mouse from 12:05 to 12:45
    afk_events = [
        {"timestamp": "2026-11-05T12:00:00Z", "duration": 300.0, "data": {"status": "not-afk"}},
        {"timestamp": "2026-11-05T12:05:00Z", "duration": 2400.0, "data": {"status": "afk"}},
    ]

    # Activity: User watching a movie in browser from 12:05 to 12:35 (1800s)
    window_events = [
        {
            "timestamp": "2026-11-05T12:05:00Z",
            "duration": 1800.0,
            "data": {"app": "Firefox", "title": "Full Movie HD - YouTube"},
        },
        # Followed by idle away period 12:35 to 12:45 (600s)
        {
            "timestamp": "2026-11-05T12:35:00Z",
            "duration": 600.0,
            "data": {"app": "Firefox", "title": "Desktop / Blank"},
        },
    ]

    # Web tab watcher records audible: True for the movie from 12:05 to 12:35
    audible_events = [
        {
            "timestamp": "2026-11-05T12:05:00Z",
            "duration": 1800.0,
            "data": {
                "audible": True,
                "url": "https://www.youtube.com/watch?v=movie",
                "title": "Full Movie HD - YouTube",
            },
        }
    ]

    # 1. WITHOUT audible support: entire 12:05-12:45 is AFK, movie is dropped
    filtered_no_audible = filter_events_by_afk(window_events, afk_events, include_audible=False)
    assert len(filtered_no_audible) == 0

    # 2. WITH audible support: movie period 12:05-12:35 is preserved as active!
    filtered_with_audible = filter_events_by_afk(
        window_events, afk_events, audible_events=audible_events, include_audible=True
    )
    assert len(filtered_with_audible) == 1
    assert filtered_with_audible[0]["data"]["title"] == "Full Movie HD - YouTube"
    assert filtered_with_audible[0]["duration"] == 1800.0


def test_filter_events_by_afk_with_audible_music_still_afk():
    # User is AFK according to keyboard/mouse from 12:05 to 12:45
    afk_events = [
        {"timestamp": "2026-11-05T12:00:00Z", "duration": 300.0, "data": {"status": "not-afk"}},
        {"timestamp": "2026-11-05T12:05:00Z", "duration": 2400.0, "data": {"status": "afk"}},
    ]

    # Activity: Browser open with Spotify playing while user went away
    window_events = [
        {
            "timestamp": "2026-11-05T12:05:00Z",
            "duration": 1800.0,
            "data": {"app": "Firefox", "title": "Spotify - Web Player"},
        }
    ]

    audible_events = [
        {
            "timestamp": "2026-11-05T12:05:00Z",
            "duration": 1800.0,
            "data": {
                "audible": True,
                "url": "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M",
                "title": "Today's Top Hits - Spotify",
            },
        }
    ]

    # Even with include_audible=True, Spotify must NOT exempt from AFK
    filtered = filter_events_by_afk(
        window_events, afk_events, audible_events=audible_events, include_audible=True
    )
    assert len(filtered) == 0


def test_find_web_buckets_for_source(monkeypatch):
    client = AWClient(base_url="http://mock-server:5600")

    mock_buckets = {
        "aw-watcher-window_fedora": {"hostname": "fedora", "type": "currentwindow"},
        "aw-watcher-web-firefox_fedora": {"hostname": "fedora", "type": "web.tab.current"},
        "aw-watcher-afk_fedora": {"hostname": "fedora", "type": "afkstatus"},
        "aw-watcher-window_laptop": {"hostname": "laptop", "type": "currentwindow"},
        "aw-watcher-web-firefox_laptop": {"hostname": "laptop", "type": "web.tab.current"},
        "aw-watcher-web-brave_laptop": {"hostname": "laptop", "type": "web.tab.current"},
        "aw-watcher-afk_laptop": {"hostname": "laptop", "type": "afkstatus"},
        "aw-watcher-android-synced-from-oneplus_12": {"hostname": "oneplus_12", "type": "currentwindow"},
        "aw-watcher-android-web-synced-from-oneplus_12": {"hostname": "oneplus_12", "type": "web.tab.current"},
        "aw-watcher-web-brave_classified": {"hostname": "classified", "type": "web.tab.current"},
        "aw-watcher-web-firefox_classified": {"hostname": "classified", "type": "web.tab.current"},
    }

    monkeypatch.setattr(client, "list_buckets", lambda: mock_buckets)

    # fedora window -> fedora web only
    assert client.find_web_buckets_for_source("aw-watcher-window_fedora") == ["aw-watcher-web-firefox_fedora"]
    # laptop window -> both laptop web buckets
    assert sorted(client.find_web_buckets_for_source("aw-watcher-window_laptop")) == [
        "aw-watcher-web-brave_laptop",
        "aw-watcher-web-firefox_laptop",
    ]
    # phone window -> phone web only
    assert client.find_web_buckets_for_source("aw-watcher-android-synced-from-oneplus_12") == [
        "aw-watcher-android-web-synced-from-oneplus_12"
    ]



def test_bucket_summary_is_incremental(monkeypatch):
    client = AWClient(base_url="http://aw")
    stored = [{"data": {"classified_by": "gemma_e2b (x)"}}, {"data": {"classified_by": "rules"}}]
    calls = {"full": 0}

    def fake_get_events(bucket_id, limit=100, start=None, end=None):
        calls["full"] += 1
        return list(stored)

    class Resp:
        status_code = 200
        def __init__(self, v): self.v = v
        def json(self): return self.v

    def fake_request(method, path, **kw):
        if path.endswith("/events/count"):
            return Resp(len(stored))
        stored.extend(kw["json"])  # POST events
        return Resp([])

    monkeypatch.setattr(client, "get_events", fake_get_events)
    monkeypatch.setattr(client, "_request", fake_request)

    s = client.get_bucket_summary("t")
    assert (s["total_events"], s["gemma_oracle"], s["fast_classifier"]) == (2, 1, 1)
    client.insert_events("t", [{"data": {"classified_by": "human_correction_memory"}}])
    s = client.get_bucket_summary("t")
    assert (s["total_events"], s["human_override"]) == (3, 1)
    assert calls["full"] == 1  # no re-download after insert

    stored.pop()  # changed outside the classifier -> count mismatch -> full refresh
    assert client.get_bucket_summary("t")["total_events"] == 2
    assert calls["full"] == 2
