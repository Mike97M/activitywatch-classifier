"""AFK (Away From Keyboard) filtering and interval intersection for ActivityWatch events."""

from datetime import datetime, timedelta, timezone
import logging
import re
from typing import Any, Dict, List, Tuple, Union

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


def filter_events_by_afk(
    events: List[Dict[str, Any]],
    afk_events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Filter raw activity events against raw AFK bucket events, keeping only non-AFK activities."""
    if not events:
        return []
    if not afk_events:
        logger.warning("Empty AFK events list provided; cannot perform AFK filtering.")
        return events

    # Verify that the stream actually contains AFK status indicators
    has_afk_status = any(
        ev.get("data", {}).get("status") in ("not-afk", "afk")
        for ev in afk_events
    )
    if not has_afk_status:
        logger.warning(
            "AFK events list does not contain 'status' ('not-afk' or 'afk'); skipping AFK filtering."
        )
        return events

    not_afk = [
        ev for ev in afk_events
        if ev.get("data", {}).get("status") == "not-afk"
    ]

    return filter_period_intersect(events, not_afk)
