"""ActivityWatch Client with endpoint auto-detection and non-destructive operations."""

import logging
from typing import Any, Dict, List, Optional
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
