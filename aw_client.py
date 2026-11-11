"""ActivityWatch Client with endpoint auto-detection and non-destructive operations."""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import httpx

from config import (
    AW_CANDIDATE_URLS,
    DEFAULT_CLIENT_NAME,
    DEFAULT_TARGET_BUCKET_ID,
    DEFAULT_TARGET_BUCKET_TYPE,
)

logger = logging.getLogger("aw_client")


class AWClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url
        self._headers = {"Host": "localhost:5600", "User-Agent": "aw-classifier/1.0"}
        self._bucket_summary_cache: Dict[str, Any] = {}
        if not self.base_url:
            self.auto_detect_server()

    def auto_detect_server(self) -> Optional[str]:
        """Try candidates to locate the active ActivityWatch server."""
        for url in AW_CANDIDATE_URLS:
            try:
                with httpx.Client(timeout=2.0) as client:
                    resp = client.get(f"{url}/api/0/info", headers=self._headers)
                    if resp.status_code == 200:
                        self.base_url = url
                        logger.info(f"Successfully connected to ActivityWatch server at {url}")
                        return url
            except Exception:
                continue
        logger.warning("Could not auto-detect ActivityWatch server from candidate URLs.")
        if not self.base_url and AW_CANDIDATE_URLS:
            self.base_url = AW_CANDIDATE_URLS[0]
        return self.base_url

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Execute request with custom headers to comply with Host validation."""
        headers = dict(self._headers)
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))
        url = f"{self.base_url.rstrip('/')}{path}"
        with httpx.Client(timeout=30.0) as client:
            resp = client.request(method, url, headers=headers, **kwargs)
            resp.raise_for_status()
            return resp

    def get_info(self) -> Dict[str, Any]:
        """Fetch ActivityWatch server status and info."""
        try:
            resp = self._request("GET", "/api/0/info")
            return resp.json()
        except Exception as e:
            # Try auto-detecting again in case server host shifted
            new_url = self.auto_detect_server()
            if new_url and new_url != self.base_url:
                resp = self._request("GET", "/api/0/info")
                return resp.json()
            raise RuntimeError(f"Failed to communicate with ActivityWatch at {self.base_url}: {e}")

    def list_buckets(self) -> Dict[str, Dict[str, Any]]:
        """List all buckets in ActivityWatch."""
        resp = self._request("GET", "/api/0/buckets")
        return resp.json()

    def list_afk_buckets(self) -> Dict[str, Dict[str, Any]]:
        """Return all buckets associated with AFK status tracking."""
        buckets = self.list_buckets()
        afk_buckets = {}
        for bid, binfo in buckets.items():
            btype = binfo.get("type", "")
            if btype == "afkstatus" or "afk" in bid.lower():
                afk_buckets[bid] = binfo
        return afk_buckets

    def find_afk_bucket_for_source(self, source_bucket_id: str) -> Optional[str]:
        """Auto-detect the matching AFK bucket for a given source bucket.

        Matches by hostname attribute, bucket name suffix (e.g. '_fedora'),
        or falls back to the solitary AFK bucket if only one is registered.
        """
        all_buckets = self.list_buckets()
        source_info = all_buckets.get(source_bucket_id, {})
        source_host = source_info.get("hostname", "")

        # Extract suffix if present (e.g. 'aw-watcher-window_fedora' -> 'fedora')
        source_suffix = source_bucket_id.split("_")[-1] if "_" in source_bucket_id else ""

        afk_candidates: List[Tuple[str, Dict[str, Any]]] = []
        for bid, binfo in all_buckets.items():
            if binfo.get("type") == "afkstatus" or "afk" in bid.lower():
                afk_candidates.append((bid, binfo))

        if not afk_candidates:
            logger.info("No AFK buckets found on ActivityWatch server.")
            return None

        # Priority 1: Match by exact hostname
        if source_host:
            for bid, binfo in afk_candidates:
                if binfo.get("hostname") == source_host:
                    return bid

        # Priority 2: Match by suffix (e.g. both end in '_fedora')
        if source_suffix:
            for bid, binfo in afk_candidates:
                if bid.endswith(f"_{source_suffix}"):
                    return bid

        # Priority 3: Exact name replacement
        candidate_name = source_bucket_id.replace("window", "afk").replace("web", "afk")
        for bid, _ in afk_candidates:
            if bid == candidate_name:
                return bid

        # Never fall back to another device's AFK bucket (e.g. do not filter phone activity using laptop AFK)
        logger.info(
            f"No matching AFK bucket found for host '{source_host}' / suffix '{source_suffix}' on {source_bucket_id}."
        )
        return None

    def find_web_buckets_for_source(self, source_bucket_id: str) -> List[str]:
        """Find all browser/web watcher buckets associated with the same device/hostname as source_bucket_id.

        Matches by hostname attribute, bucket name suffix (e.g. '_fedora', '_laptop'),
        or self if source_bucket_id is itself a web bucket. Avoids cross-device contamination.
        """
        all_buckets = self.list_buckets()
        source_info = all_buckets.get(source_bucket_id, {})
        source_host = source_info.get("hostname", "")
        source_suffix = source_bucket_id.split("_")[-1] if "_" in source_bucket_id else ""

        web_candidates: List[str] = []
        for bid, binfo in all_buckets.items():
            btype = binfo.get("type", "")
            # Identify web watcher buckets (excluding classifier generated or afk buckets)
            is_web = (
                (btype == "web.tab.current" or "web" in bid.lower())
                and "afk" not in bid.lower()
                and "classified" not in bid.lower()
                and binfo.get("hostname") != "classified"
            )
            if not is_web:
                continue

            b_host = binfo.get("hostname", "")
            b_suffix = bid.split("_")[-1] if "_" in bid else ""

            if source_host and b_host and source_host == b_host:
                web_candidates.append(bid)
            elif source_suffix and b_suffix and source_suffix == b_suffix:
                web_candidates.append(bid)
            elif not source_host and not source_suffix and bid == source_bucket_id:
                web_candidates.append(bid)

        return sorted(list(set(web_candidates)))

    def get_bucket(self, bucket_id: str) -> Optional[Dict[str, Any]]:
        """Get details for a single bucket."""
        try:
            resp = self._request("GET", f"/api/0/buckets/{bucket_id}")
            return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def create_bucket(
        self,
        bucket_id: str = DEFAULT_TARGET_BUCKET_ID,
        bucket_type: str = DEFAULT_TARGET_BUCKET_TYPE,
        client_name: str = DEFAULT_CLIENT_NAME,
        hostname: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new bucket without modifying existing buckets."""
        if not hostname:
            try:
                info = self.get_info()
                hostname = info.get("hostname", "localhost")
            except Exception:
                hostname = "localhost"

        payload = {
            "client": client_name,
            "type": bucket_type,
            "hostname": hostname,
        }

        # Check if already exists
        existing = self.get_bucket(bucket_id)
        if existing:
            logger.info(f"Bucket '{bucket_id}' already exists.")
            return existing

        resp = self._request("POST", f"/api/0/buckets/{bucket_id}", json=payload)
        logger.info(f"Created new bucket: '{bucket_id}'")
        return {"id": bucket_id, "created": True, "details": payload}

    def get_events(
        self,
        bucket_id: str,
        limit: int = 100,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch raw events from a source bucket (never modified in place)."""
        params = {}
        if limit and limit > 0:
            params["limit"] = limit
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        resp = self._request("GET", f"/api/0/buckets/{bucket_id}/events", params=params)
        return resp.json()

    def get_existing_target_timestamps(
        self,
        bucket_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Set[str]:
        """Fetch timestamps of events already present in target bucket for non-destructive deduplication."""
        try:
            events = self.get_events(bucket_id, limit=0, start=start, end=end)
            return {str(ev["timestamp"]) for ev in events if ev.get("timestamp")}
        except Exception as e:
            logger.debug(f"Could not fetch existing timestamps for '{bucket_id}': {e}")
            return set()

    def insert_events(self, bucket_id: str, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Insert events into the designated target bucket."""
        if not events:
            return []
        if hasattr(self, "_bucket_summary_cache"):
            self._bucket_summary_cache.pop(bucket_id, None)
        # Chunk insert in batches of 100 to avoid request entity too large errors
        chunk_size = 100
        inserted_results = []
        for i in range(0, len(events), chunk_size):
            chunk = events[i : i + chunk_size]
            resp = self._request("POST", f"/api/0/buckets/{bucket_id}/events", json=chunk)
            if resp.status_code in (200, 201):
                inserted_results.extend(resp.json())
        return inserted_results

    def get_bucket_summary(self, bucket_id: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Summarize classified activities in target bucket: total, fast_classifier, gemma, cached, etc."""
        now = time.time()
        cache_entry = getattr(self, "_bucket_summary_cache", {}).get(bucket_id)
        if not force_refresh and cache_entry and (now - cache_entry.get("time", 0) < 10.0):
            return cache_entry["data"]

        try:
            events = self.get_events(bucket_id, limit=-1)
        except Exception as e:
            logger.debug(f"Could not fetch events for summary of '{bucket_id}': {e}")
            events = []

        fast_count = 0
        gemma_count = 0
        cached_count = 0
        human_count = 0

        for e in events:
            data = e.get("data", {}) if isinstance(e, dict) else {}
            by = str(data.get("classified_by", "")).lower()
            if "human" in by or "override" in by:
                human_count += 1
            elif "cache" in by:
                cached_count += 1
            elif "gemma" in by or "oracle" in by or "gemini" in by or "llm" in by:
                gemma_count += 1
            elif "fast" in by:
                fast_count += 1
            elif by:
                fast_count += 1

        summary = {
            "bucket_id": bucket_id,
            "total_events": len(events),
            "fast_classifier": fast_count,
            "gemma_oracle": gemma_count,
            "cached_memory": cached_count,
            "human_override": human_count,
        }

        if not hasattr(self, "_bucket_summary_cache"):
            self._bucket_summary_cache = {}
        self._bucket_summary_cache[bucket_id] = {"time": now, "data": summary}
        return summary

    def delete_bucket(self, bucket_id: str, force: bool = False) -> bool:
        """Delete a bucket (useful for resetting the target bucket)."""
        if hasattr(self, "_bucket_summary_cache"):
            self._bucket_summary_cache.pop(bucket_id, None)
        try:
            params = {"force": "1"} if force else {}
            resp = self._request("DELETE", f"/api/0/buckets/{bucket_id}", params=params)
            return resp.status_code in (200, 204)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return False
            logger.warning(f"Failed to delete bucket '{bucket_id}': {e}")
            return False
        except Exception as e:
            logger.warning(f"Failed to delete bucket '{bucket_id}': {e}")
            return False

    def get_setting(self, key: str) -> Any:
        """Fetch a specific setting from ActivityWatch server."""
        try:
            resp = self._request("GET", f"/api/0/settings/{key}")
            return resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch setting '{key}': {e}")
            return None

    def set_setting(self, key: str, value: Any) -> bool:
        """Update a specific setting in ActivityWatch server."""
        try:
            resp = self._request("POST", f"/api/0/settings/{key}", json=value)
            return resp.status_code in (200, 201)
        except Exception as e:
            logger.error(f"Failed to save setting '{key}' to ActivityWatch: {e}")
            return False

    def sync_classified_host_bucket(
        self,
        events: List[Dict[str, Any]],
        host: str = "classified",
    ) -> bool:
        """Synchronize classified activities into ActivityWatch currentwindow, web.tab.current, and afkstatus buckets.
        
        This enables viewing the classified events directly under the 'Activity' tab
        of ActivityWatch (e.g. http://localhost:5600/#/activity/classified/), including
        Top Applications, Window Titles, Top Domains, Top URLs, and Browser Tabs.
        """
        if not events:
            return True

        from datetime import timedelta
        from urllib.parse import urlparse
        from afk_filter import parse_iso_timestamp, to_iso_timestamp

        window_bucket = f"aw-watcher-window_{host}"
        afk_bucket = f"aw-watcher-afk_{host}"

        # 1. Ensure core buckets exist
        self.create_bucket(bucket_id=window_bucket, bucket_type="currentwindow", hostname=host)
        self.create_bucket(bucket_id=afk_bucket, bucket_type="afkstatus", hostname=host)

        def detect_browser(data: Dict[str, Any]) -> Tuple[Optional[str], str]:
            app = str(data.get("app") or "").strip()
            src = str(data.get("source_bucket") or "").strip().lower()
            url = str(data.get("url") or "").strip()
            combined = f"{src} {app}".lower()

            if "brave" in combined:
                return "brave", "Brave-browser"
            elif any(f in combined for f in ("firefox", "librewolf", "waterfox", "nightly")):
                return "firefox", "Firefox"
            elif "edge" in combined or "msedge" in combined:
                return "edge", "Microsoft-edge"
            elif "opera" in combined:
                return "opera", "Opera"
            elif "vivaldi" in combined:
                return "vivaldi", "Vivaldi"
            elif "zen" in combined:
                return "zen", "Zen"
            elif "floorp" in combined:
                return "floorp", "Floorp"
            elif "arc" in combined:
                return "arc", "Arc"
            elif "chrome" in combined or "chromium" in combined:
                return "chrome", "Google-chrome"
            elif url:
                return "chrome", "Google-chrome"
            else:
                return None, app or "Unknown"

        # 2. Separate into window events and browser web tab events partitioned by browser slug
        window_events = []
        web_events_by_browser: Dict[str, List[Dict[str, Any]]] = {}

        for e in events:
            d = dict(e.get("data", {}))
            url_val = str(d.get("url") or "").strip()
            app_val = str(d.get("app") or "").strip()

            browser_slug, canonical_app = detect_browser(d)

            # Ensure window events for web activities carry an app name recognized by ActivityWatch
            if url_val and browser_slug:
                if not app_val or app_val.lower() in ("browser", "unknown", "web") or browser_slug not in app_val.lower():
                    d["app"] = canonical_app
                elif app_val.lower() == browser_slug:
                    d["app"] = canonical_app
            elif not app_val:
                d["app"] = canonical_app

            orig_title = d.get("title") or d.get("app") or "Unknown"
            if not d.get("title"):
                d["title"] = orig_title

            if "category" in d and "$category" not in d:
                d["$category"] = [d.get("category", "Uncategorized"), d.get("subcategory", "General Activity")]

            # Bridge ActivityWatch's native architecture limitation: ActivityWatch native Category Tree
            # only categorizes window titles from aw-watcher-window. When a browser event carries a URL,
            # append the domain tag so native regex rules matching domains/URLs match directly.
            if url_val and browser_slug:
                try:
                    test_url = url_val if "://" in url_val else f"https://{url_val}"
                    parsed_netloc = urlparse(test_url).netloc.split(":")[0].lower()
                    if parsed_netloc and parsed_netloc not in d["title"].lower():
                        d["title"] = f"{d['title']} [{parsed_netloc}]"
                except Exception:
                    pass

            window_events.append({
                "timestamp": e.get("timestamp"),
                "duration": e.get("duration", 0.0),
                "data": d,
            })

            if url_val and browser_slug:
                if browser_slug not in web_events_by_browser:
                    web_events_by_browser[browser_slug] = []
                web_events_by_browser[browser_slug].append({
                    "timestamp": e.get("timestamp"),
                    "duration": e.get("duration", 0.0),
                    "data": {
                        "title": orig_title or url_val,
                        "url": url_val,
                        "audible": d.get("audible", False),
                        "incognito": d.get("incognito", False),
                        "tabCount": d.get("tabCount", 1),
                        "category": d.get("category"),
                        "subcategory": d.get("subcategory"),
                        "full_category": d.get("full_category"),
                        "confidence": d.get("confidence", 1.0),
                        "classified_by": d.get("classified_by", ""),
                    },
                })

        self.insert_events(window_bucket, window_events)

        # 3. Synchronize browser web events into browser-specific buckets
        # (e.g. aw-watcher-web-brave_classified, aw-watcher-web-firefox_classified)
        for b_slug, b_events in web_events_by_browser.items():
            if not b_events:
                continue
            web_bucket = f"aw-watcher-web-{b_slug}_{host}"
            self.create_bucket(
                bucket_id=web_bucket,
                bucket_type="web.tab.current",
                client_name="aw-watcher-web",
                hostname=host,
            )
            self.insert_events(web_bucket, b_events)
            logger.info(f"Synchronized {len(b_events)} web events with URLs to '{web_bucket}'")

        # 4. Clean up legacy unpartitioned bucket aw-watcher-web_{host} if present
        legacy_web_bucket = f"aw-watcher-web_{host}"
        try:
            self.delete_bucket(legacy_web_bucket)
        except Exception:
            pass

        # 5. Insert non-AFK span covering the events so ActivityWatch marks them active
        try:
            min_time = min(parse_iso_timestamp(e["timestamp"]) for e in events)
            max_time = max(
                parse_iso_timestamp(e["timestamp"]) + timedelta(seconds=float(e.get("duration", 0.0)))
                for e in events
            )
            span_dur = max(60.0, (max_time - min_time).total_seconds() + 600.0)
            afk_span = [{
                "timestamp": to_iso_timestamp(min_time - timedelta(minutes=5)),
                "duration": round(span_dur, 2),
                "data": {"status": "not-afk"},
            }]
            self.insert_events(afk_bucket, afk_span)
        except Exception as e:
            logger.warning(f"Failed to insert AFK active span for host '{host}': {e}")

        logger.info(f"Synchronized {len(window_events)} events to Activity tab host '{host}'")
        return True
