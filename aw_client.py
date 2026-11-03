"""ActivityWatch Client with endpoint auto-detection and non-destructive operations."""

import logging
from typing import Any, Dict, List, Optional, Tuple
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

        # Priority 4: If only 1 AFK bucket exists on server, use it
        if len(afk_candidates) == 1:
            return afk_candidates[0][0]

        # Priority 5: Return first candidate as fallback
        return afk_candidates[0][0]

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

    def insert_events(self, bucket_id: str, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Insert events into the designated target bucket."""
        if not events:
            return []
        # Chunk insert in batches of 100 to avoid request entity too large errors
        chunk_size = 100
        inserted_results = []
        for i in range(0, len(events), chunk_size):
            chunk = events[i : i + chunk_size]
            resp = self._request("POST", f"/api/0/buckets/{bucket_id}/events", json=chunk)
            if resp.status_code in (200, 201):
                inserted_results.extend(resp.json())
        return inserted_results

    def delete_bucket(self, bucket_id: str) -> bool:
        """Delete a bucket (useful for resetting the target bucket)."""
        try:
            self._request("DELETE", f"/api/0/buckets/{bucket_id}")
            return True
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
        """Synchronize classified activities into ActivityWatch currentwindow & afkstatus buckets.
        
        This enables viewing the classified events directly under the 'Activity' tab
        of ActivityWatch (e.g. http://localhost:5600/#/activity/classified/).
        """
        if not events:
            return True

        from datetime import timedelta
        from afk_filter import parse_iso_timestamp, to_iso_timestamp

        window_bucket = f"aw-watcher-window_{host}"
        afk_bucket = f"aw-watcher-afk_{host}"

        # 1. Ensure buckets exist
        self.create_bucket(bucket_id=window_bucket, bucket_type="currentwindow", hostname=host)
        self.create_bucket(bucket_id=afk_bucket, bucket_type="afkstatus", hostname=host)

        # 2. Insert window events
        window_events = []
        for e in events:
            d = dict(e.get("data", {}))
            window_events.append({
                "timestamp": e.get("timestamp"),
                "duration": e.get("duration", 0.0),
                "data": d,
            })
        self.insert_events(window_bucket, window_events)

        # 3. Insert non-AFK span covering the events so ActivityWatch marks them active
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
