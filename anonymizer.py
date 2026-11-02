"""PII & Sensitive Data Anonymizer for ActivityWatch window titles, URLs, and features.

Removes usernames, private file paths, emails, IP addresses, sensitive tokens/keys,
and personal contact names before saving training data or publishing to Git.
"""

import getpass
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


class TextAnonymizer:
    """Sanitizes text fields to ensure zero PII leaks into training datasets or Git."""

    def __init__(self, extra_usernames: Optional[List[str]] = None):
        self.usernames: Set[str] = set()

        # Detect system usernames dynamically
        try:
            self.usernames.add(getpass.getuser().lower())
        except Exception:
            pass

        for env_var in ["USER", "LOGNAME", "USERNAME"]:
            val = os.environ.get(env_var)
            if val:
                self.usernames.add(val.lower())

        try:
            home_name = Path.home().name.lower()
            if home_name:
                self.usernames.add(home_name)
        except Exception:
            pass

        if extra_usernames:
            for u in extra_usernames:
                if u:
                    self.usernames.add(u.lower())

        # Discard generic terms that shouldn't be blindly masked
        self.usernames.discard("root")
        self.usernames.discard("user")
        self.usernames.discard("guest")

        # Compile regular expressions for efficiency
        self._email_re = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        )
        self._ipv4_re = re.compile(
            r"\b(?!(?:127\.0\.0\.1|0\.0\.0\.0)\b)(?:\d{1,3}\.){3}\d{1,3}\b"
        )
        self._phone_re = re.compile(
            r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        )
        self._uuid_re = re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        )
        # Linux / Unix user prompt pattern: e.g. "jdoe@laptop:~" -> "<USER>@<HOST>:~"
        self._shell_prompt_re = re.compile(
            r"\b([a-zA-Z0-9_.-]+)@([a-zA-Z0-9_.-]+):([~/\w.-]*)"
        )
        # Unix paths: /home/<user>/... -> ~/<PATH>
        self._home_path_re = re.compile(
            r"(?:/home|/Users)/[a-zA-Z0-9_.-]+(/[\w.-]+)*"
        )
        # Windows paths: C:\Users\<user>\... -> C:\Users\<USER>\<PATH>
        self._win_path_re = re.compile(
            r"[A-Za-z]:\\Users\\[a-zA-Z0-9_.-]+(\\[\w.-]+)*", re.IGNORECASE
        )
        # Chat / Contact patterns: e.g. "Direct Message - Alice Smith" or "Chat with Bob"
        self._dm_re = re.compile(
            r"(?i)\b(direct messages?|dm with|chat with|conversation with)\s*[-:]?\s*([A-Za-z0-9_.\s]{2,30})"
        )

        self._sensitive_param_names = {
            "token", "key", "auth", "api_key", "apikey", "code", "access_token",
            "refresh_token", "secret", "password", "passwd", "session", "session_id",
            "sig", "signature", "jwt", "id_token", "credential"
        }

    def sanitize_title(self, title: str) -> str:
        """Sanitize window title removing paths, usernames, prompt info, emails, and PII."""
        if not title:
            return ""

        text = title.strip()

        # 1. Sanitize shell prompts (e.g. jdoe@laptop:~ -> <USER>@<HOST>:~)
        text = self._shell_prompt_re.sub(r"<USER>@\2:\3", text)

        # 2. Sanitize home directories
        text = self._home_path_re.sub(r"~/<PATH>", text)
        text = self._win_path_re.sub(r"C:\\Users\\<USER>\\<PATH>", text)

        # 3. Sanitize emails
        text = self._email_re.sub("<EMAIL>", text)

        # 4. Sanitize external IPs
        text = self._ipv4_re.sub("<IP>", text)

        # 5. Sanitize UUIDs / GUIDs
        text = self._uuid_re.sub("<UUID>", text)

        # 6. Sanitize phone numbers
        text = self._phone_re.sub("<PHONE>", text)

        # 7. Sanitize chat contacts
        text = self._dm_re.sub(r"\1 - <CONTACT>", text)

        # 8. Mask detected usernames case-insensitively
        for uname in self.usernames:
            if len(uname) >= 3:
                pattern = re.compile(rf"\b{re.escape(uname)}\b", re.IGNORECASE)
                text = pattern.sub("<USER>", text)

        return text

    def sanitize_url(self, url: str) -> str:
        """Sanitize URLs by stripping sensitive query params, tokens, and basic auth credentials."""
        if not url:
            return ""

        try:
            parsed = urlparse(url.strip())
            if not parsed.scheme or not parsed.netloc:
                # Not a full URL, treat as general text
                return self.sanitize_title(url)

            # Strip basic auth: http://user:pass@domain -> http://domain
            netloc = parsed.netloc
            if "@" in netloc:
                netloc = netloc.split("@")[-1]

            # Sanitize IP address in host if it's not localhost
            if self._ipv4_re.search(netloc):
                netloc = self._ipv4_re.sub("<IP>", netloc)

            # Sanitize query parameters
            sanitized_query = ""
            if parsed.query:
                pairs = parse_qsl(parsed.query, keep_blank_values=True)
                clean_pairs = []
                for k, v in pairs:
                    if k.lower() in self._sensitive_param_names or len(v) > 40:
                        clean_pairs.append((k, "<REDACTED>"))
                    elif self._email_re.search(v):
                        clean_pairs.append((k, "<EMAIL>"))
                    else:
                        clean_pairs.append((k, v))
                sanitized_query = urlencode(clean_pairs)

            # Rebuild clean URL
            clean_url = urlunparse((
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                sanitized_query,
                "",  # remove fragment hash which often contains tokens
            ))
            return clean_url
        except Exception:
            return self.sanitize_title(url)

    def sanitize_event_data(self, data: Dict[str, any]) -> Dict[str, any]:
        """Produce clean event payload free of PII."""
        cleaned = dict(data)
        if "title" in cleaned:
            cleaned["title"] = self.sanitize_title(str(cleaned["title"]))
        if "url" in cleaned:
            cleaned["url"] = self.sanitize_url(str(cleaned["url"]))
        return cleaned


# Global singleton instance
anonymizer = TextAnonymizer()
