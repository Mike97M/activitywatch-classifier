"""AFK (Away From Keyboard) filtering and interval intersection for ActivityWatch events."""

from datetime import datetime, timedelta, timezone
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("afk_filter")


def parse_iso_timestamp(ts: Union[str, datetime]) -> datetime:
    """Parse ISO-8601 timestamp string with timezone support and nanosecond tolerance."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)

    if not isinstance(ts, str):
        raise ValueError(f"Expected ISO timestamp string or datetime, got {type(ts)}")

    # Fast path: standard ISO 8601
    try:
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass

    # Normalize trailing Z to +00:00
    ts_clean = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
    try:
        dt = datetime.fromisoformat(ts_clean)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        # Fallback for irregular fractional digits (e.g. > 6 decimal places on older platforms)
        match = re.match(
            r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})(\.\d+)?([+-]\d{2}:?\d{2})?$",
            ts_clean,
        )
        if match:
            base, frac, tz = match.groups()
            frac = (frac or ".0")[:7]  # . + 6 digits max for python microsecond
            tz = tz or "+00:00"
            if len(tz) == 5 and ":" not in tz:
                tz = tz[:3] + ":" + tz[3:]
            iso_str = f"{base.replace(' ', 'T')}{frac}{tz}"
            return datetime.fromisoformat(iso_str)
        raise ValueError(f"Cannot parse timestamp: '{ts}'")


def to_iso_timestamp(dt: datetime) -> str:
    """Convert datetime to ActivityWatch-compatible ISO-8601 string ending in Z."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def resolve_time_range(
    time_range: Optional[str] = None,
    start_time: Optional[Union[str, datetime]] = None,
    end_time: Optional[Union[str, datetime]] = None,
    now: Optional[datetime] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Convert time range specification into ActivityWatch ISO start and end timestamps.

    Supported time_range values:
    - 'today': From midnight today (UTC) to now
    - 'last_day' / '24h' / 'day': Past 24 hours
    - 'last_week' / '7d' / 'week': Past 7 days
    - 'last_month' / '30d' / 'month': Past 30 days
    - 'all': None (unbounded)
    - Custom start_time / end_time: Parsed and formatted to ISO strings
    """
    curr = now or datetime.now(timezone.utc)
    if curr.tzinfo is None:
        curr = curr.replace(tzinfo=timezone.utc)

    # 1. Explicit start / end override
    if start_time or end_time:
        start_iso = to_iso_timestamp(parse_iso_timestamp(start_time)) if start_time else None
        end_iso = to_iso_timestamp(parse_iso_timestamp(end_time)) if end_time else None
        return (start_iso, end_iso)

    if not time_range or time_range.strip().lower() in ("count", "none", "events"):
        return (None, None)

    tr = time_range.strip().lower()
    if tr in ("today", "current_day"):
        start_dt = curr.replace(hour=0, minute=0, second=0, microsecond=0)
        return (to_iso_timestamp(start_dt), to_iso_timestamp(curr))
    elif tr in ("last_day", "24h", "1d", "day"):
        start_dt = curr - timedelta(days=1)
        return (to_iso_timestamp(start_dt), to_iso_timestamp(curr))
    elif tr in ("last_week", "7d", "1w", "week"):
        start_dt = curr - timedelta(days=7)
        return (to_iso_timestamp(start_dt), to_iso_timestamp(curr))
    elif tr in ("last_month", "30d", "1m", "month"):
        start_dt = curr - timedelta(days=30)
        return (to_iso_timestamp(start_dt), to_iso_timestamp(curr))
    elif tr in ("all", "all_time"):
        return (None, None)

    return (None, None)


def merge_intervals(intervals: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    """Merge contiguous or overlapping (start, end) datetime intervals."""
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[datetime, datetime]] = []

    for start, end in sorted_intervals:
        if end <= start:
            continue
        if not merged:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

    return merged


def filter_period_intersect(
    events: List[Dict[str, Any]],
    not_afk_events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Intersect activity events with not-afk intervals.

    Returns only the portion of events where the user was actively present (not AFK).
    Events falling completely inside AFK intervals are discarded.
    Events spanning across AFK boundaries are trimmed to their active not-afk duration.
    """
    if not events:
        return []
    if not not_afk_events:
        logger.warning("No not-afk intervals provided. Discarding events under strict AFK guard.")
        return []

    # 1. Parse not-afk intervals
    raw_not_afk: List[Tuple[datetime, datetime]] = []
    for na in not_afk_events:
        try:
            start = parse_iso_timestamp(na["timestamp"])
            dur = max(0.0, float(na.get("duration", 0.0)))
            end = start + timedelta(seconds=dur)
            raw_not_afk.append((start, end))
        except Exception as e:
            logger.debug(f"Skipping unparseable not-afk event: {e}")
            continue

    merged_not_afk = merge_intervals(raw_not_afk)
    if not merged_not_afk:
        return []

    # 2. Parse source events
    parsed_events = []
    for e in events:
        try:
            start = parse_iso_timestamp(e["timestamp"])
            dur = max(0.0, float(e.get("duration", 0.0)))
            end = start + timedelta(seconds=dur)
            parsed_events.append((start, end, dur, e))
        except Exception as e:
            logger.debug(f"Skipping unparseable event: {e}")
            continue

    parsed_events.sort(key=lambda x: x[0])

    # 3. Two-pointer sweep intersection
    results: List[Dict[str, Any]] = []
    j = 0
    num_na = len(merged_not_afk)

    for e_start, e_end, e_dur, raw_e in parsed_events:
        # Advance pointer to first candidate not_afk interval
        while j < num_na and merged_not_afk[j][1] <= e_start:
            j += 1

        # Point events (duration == 0)
        if e_dur == 0.0:
            k = j
            while k < num_na and merged_not_afk[k][0] <= e_start:
                if merged_not_afk[k][0] <= e_start <= merged_not_afk[k][1]:
                    results.append(dict(raw_e))
                    break
                k += 1
            continue

        # Duration interval events
        k = j
        while k < num_na and merged_not_afk[k][0] < e_end:
            na_start, na_end = merged_not_afk[k]
            overlap_start = max(e_start, na_start)
            overlap_end = min(e_end, na_end)
            overlap_dur = (overlap_end - overlap_start).total_seconds()

            if overlap_dur > 0.0:
                new_event = dict(raw_e)
                new_event["timestamp"] = to_iso_timestamp(overlap_start)
                new_event["duration"] = round(overlap_dur, 4)
                results.append(new_event)
            k += 1

    return results


EXCLUDED_MUSIC_DOMAINS = (
    "music.youtube.com",
    "spotify.com",
    "open.spotify.com",
    "play.spotify.com",
    "soundcloud.com",
    "deezer.com",
    "tidal.com",
    "listen.tidal.com",
    "music.apple.com",
    "bandcamp.com",
    "pandora.com",
    "music.amazon.",
    "qobuz.com",
    "tunein.com",
    "iheart.com",
    "mixcloud.com",
    "audiomack.com",
    "last.fm",
    "di.fm",
)

EXCLUDED_MUSIC_TITLE_PHRASES = (
    "youtube music",
    "spotify",
    "soundcloud",
    "deezer",
    "tidal",
    "apple music",
    "amazon music",
    "bandcamp",
    "pandora",
)

EXCLUDED_MUSIC_APPS = (
    "spotify",
    "deezer",
    "tidal",
    "apple music",
    "youtube music",
)


def is_excluded_music(data: Dict[str, Any]) -> bool:
    """Return True if activity is from a background music service (e.g. Spotify, YouTube Music).

    These background audio streams should NOT exempt the user from AFK detection,
    allowing idle/sleep periods while listening to music to properly register as AFK.
    """
    if not data:
        return False

    url = str(data.get("url") or "").lower()
    title = str(data.get("title") or "").lower()
    app = str(data.get("app") or "").lower()

    # 1. URL domain check
    if url:
        for domain in EXCLUDED_MUSIC_DOMAINS:
            if domain in url:
                return True

    # 2. Title phrase check
    if title:
        for phrase in EXCLUDED_MUSIC_TITLE_PHRASES:
            if phrase in title:
                return True

    # 3. Dedicated desktop music player check
    if app:
        for app_name in EXCLUDED_MUSIC_APPS:
            if app_name in app:
                return True

    return False


def is_audible_non_afk(event: Dict[str, Any]) -> bool:
    """Return True if an event represents active audible media (e.g. movies, videos, streams).

    Must have `audible: True` and NOT be an excluded background music service.
    """
    data = event.get("data", {})
    if data.get("audible") is not True:
        return False
    if is_excluded_music(data):
        return False
    return True


def extract_audible_non_afk_intervals(
    events: List[Dict[str, Any]],
    pulsetime: float = 65.0,
) -> List[Dict[str, Any]]:
    """Extract audible non-music events and convert them to not-afk intervals.

    ActivityWatch web watcher heartbeats typically have duration 0.0 with ~60s intervals.
    Sub-pulsetime durations are expanded to pulsetime (default 65.0s) so consecutive
    heartbeats merge seamlessly into uninterrupted media watching periods.
    """
    if not events:
        return []

    intervals: List[Dict[str, Any]] = []
    for ev in events:
        if is_audible_non_afk(ev):
            raw_dur = max(0.0, float(ev.get("duration", 0.0)))
            dur = max(raw_dur, pulsetime)
            intervals.append({
                "timestamp": ev["timestamp"],
                "duration": dur,
                "data": {
                    "status": "not-afk",
                    "audible": True,
                    "title": ev.get("data", {}).get("title"),
                    "url": ev.get("data", {}).get("url"),
                },
            })
    return intervals


def filter_events_by_afk(
    events: List[Dict[str, Any]],
    afk_events: List[Dict[str, Any]],
    audible_events: Optional[List[Dict[str, Any]]] = None,
    include_audible: bool = True,
) -> List[Dict[str, Any]]:
    """Filter raw activity events against raw AFK bucket events, keeping only non-AFK activities.

    If include_audible is True, browser tabs actively playing audio (e.g. YouTube videos,
    movies, Netflix, Twitch) are treated as active (not-afk) even without keyboard/mouse input,
    while background music services (Spotify, YouTube Music, Deezer, etc.) are excluded.
    """
    if not events:
        return []

    # 1. Collect standard not-afk intervals from AFK watcher
    not_afk: List[Dict[str, Any]] = []
    if afk_events:
        has_afk_status = any(
            ev.get("data", {}).get("status") in ("not-afk", "afk")
            for ev in afk_events
        )
        if has_afk_status:
            not_afk.extend([
                ev for ev in afk_events
                if ev.get("data", {}).get("status") == "not-afk"
            ])
        else:
            logger.warning(
                "AFK events list does not contain 'status' ('not-afk' or 'afk'); skipping AFK status filtering."
            )

    # 2. Collect audible media intervals if enabled
    if include_audible:
        if audible_events:
            not_afk.extend(extract_audible_non_afk_intervals(audible_events))
        # Also check source events in case the source itself is a browser watcher
        not_afk.extend(extract_audible_non_afk_intervals(events))

    # If no not-afk intervals exist and no afk events were provided, keep events
    if not not_afk and not afk_events:
        logger.warning("Empty AFK events list provided; cannot perform AFK filtering.")
        return events

    return filter_period_intersect(events, not_afk)
