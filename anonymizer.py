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

        # JWT tokens: header.payload.signature or 2-part tokens (case-insensitive for lowercased titles)
        self._jwt_re = re.compile(
            r"\beyJ[a-zA-Z0-9_\-]{8,}\.[a-zA-Z0-9_\-]{8,}(?:\.[a-zA-Z0-9_\-]+)?\b",
            re.IGNORECASE,
        )

        # Hexadecimal hashes / session tokens (24-64 hex chars, e.g. MD5 32, SHA1 40, SHA256 64)
        self._hex_token_re = re.compile(r"\b[0-9a-fA-F]{24,64}\b")

        # Sensitive key-value parameters appearing in URLs, titles, fragments (session=..., state=..., etc.)
        self._sensitive_kv_re = re.compile(
            r"(?i)\b("
            r"session(?:_id|_state)?|sid|token|access_token|id_token|refresh_token|"
            r"auth(?:user)?|secret|password|passwd|key|api_key|apikey|code|state|nonce|"
            r"csrf|xsrf|_csrf|phpsessid|jsessionid|aspsessionid|ticket|samlart|relaystate|"
            r"assertion|otp|totp|pin|passcode|credential|credentials|sig|signature"
            r")=([^&\s\"\'#]+)"
        )

        self._sensitive_param_names = {
            "token", "key", "auth", "api_key", "apikey", "code", "access_token",
            "refresh_token", "secret", "password", "passwd", "session", "session_id",
            "sessionid", "session_state", "sid", "sig", "signature", "jwt", "id_token",
            "credential", "credentials", "state", "nonce", "csrf", "xsrf", "_csrf",
            "ticket", "authuser", "client_id", "client_secret", "phpsessid", "jsessionid",
            "aspsessionid", "samlart", "relaystate", "assertion", "otp", "totp", "pin",
            "passcode", "user_id", "userid", "account_id",
        }

    def sanitize_title(self, title: str) -> str:
        """Sanitize window title removing paths, usernames, prompt info, emails, tokens, and PII."""
        if not title:
            return ""

        text = title.strip()

        # 1. Sanitize JWT tokens (e.g. eyJhbGciOi... or eyj0exai...)
        text = self._jwt_re.sub("<JWT>", text)

        # 2. Sanitize query and fragment sensitive key-values (session=..., access_token=..., state=..., code=...)
        text = self._sensitive_kv_re.sub(r"\1=<REDACTED>", text)

        # 3. Sanitize hexadecimal session/auth tokens (24-64 hex chars: MD5, SHA-1, etc.)
        text = self._hex_token_re.sub("<TOKEN>", text)

        # 4. Sanitize shell prompts (e.g. jdoe@laptop:~ -> <USER>@<HOST>:~)
        text = self._shell_prompt_re.sub(r"<USER>@\2:\3", text)

        # 5. Sanitize home directories
        text = self._home_path_re.sub(r"~/<PATH>", text)
        text = self._win_path_re.sub(r"C:\\Users\\<USER>\\<PATH>", text)

        # 6. Sanitize emails
        text = self._email_re.sub("<EMAIL>", text)

        # 7. Sanitize external IPs
        text = self._ipv4_re.sub("<IP>", text)

        # 8. Sanitize UUIDs / GUIDs
        text = self._uuid_re.sub("<UUID>", text)

        # 9. Sanitize phone numbers
        text = self._phone_re.sub("<PHONE>", text)

        # 10. Sanitize chat contacts
        text = self._dm_re.sub(r"\1 - <CONTACT>", text)

        # 11. Mask detected usernames case-insensitively
        for uname in self.usernames:
            if len(uname) >= 3:
                pattern = re.compile(rf"\b{re.escape(uname)}\b", re.IGNORECASE)
                text = pattern.sub("<USER>", text)

        return text

    def sanitize_text(self, text: str) -> str:
        """Sanitize generic feature string or mixed text containing titles, URLs, and metadata."""
        if not text:
            return ""
        return self.sanitize_title(text)

    def sanitize_url(self, url: str) -> str:
        """Sanitize URLs by stripping sensitive query params, tokens, path hashes, and basic auth credentials."""
        if not url:
            return ""

        try:
            raw_url = url.strip()
            has_scheme = "://" in raw_url
            test_url = raw_url if has_scheme else f"https://{raw_url}"
            parsed = urlparse(test_url)
            if not parsed.netloc:
                # Not a full URL, treat as general text
                return self.sanitize_title(url)

            # Strip basic auth: http://user:pass@domain -> http://domain
            netloc = parsed.netloc
            if "@" in netloc:
                netloc = netloc.split("@")[-1]

            # Sanitize IP address in host if it's not localhost
            if self._ipv4_re.search(netloc):
                netloc = self._ipv4_re.sub("<IP>", netloc)

            # Sanitize path segments: redact hex tokens, UUIDs, usernames, home paths
            path_segments = parsed.path.split("/")
            clean_segments = []
            for seg in path_segments:
                seg_clean = seg
                if self._hex_token_re.fullmatch(seg_clean):
                    seg_clean = "<TOKEN>"
                elif self._uuid_re.fullmatch(seg_clean):
                    seg_clean = "<UUID>"
                else:
                    seg_clean = self._uuid_re.sub("<UUID>", seg_clean)
                    seg_clean = self._hex_token_re.sub("<TOKEN>", seg_clean)
                clean_segments.append(seg_clean)
            clean_path = "/".join(clean_segments)
            for uname in self.usernames:
                if len(uname) >= 3:
                    clean_path = re.sub(
                        rf"\b{re.escape(uname)}\b", "<USER>", clean_path, flags=re.IGNORECASE
                    )

            # Sanitize query parameters
            sanitized_query = ""
            if parsed.query:
                pairs = parse_qsl(parsed.query, keep_blank_values=True)
                clean_pairs = []
                for k, v in pairs:
                    k_lower = k.lower()
                    if (
                        k_lower in self._sensitive_param_names
                        or len(v) > 24
                        or self._hex_token_re.search(v)
                        or self._jwt_re.search(v)
                    ):
                        clean_pairs.append((k, "<REDACTED>"))
                    elif self._email_re.search(v):
                        clean_pairs.append((k, "<EMAIL>"))
                    elif self._uuid_re.search(v):
                        clean_pairs.append((k, "<UUID>"))
                    else:
                        clean_pairs.append((k, v))
                sanitized_query = urlencode(clean_pairs)

            # Rebuild clean URL (dropping fragment hash which often contains tokens)
            clean_url = urlunparse((
                parsed.scheme if has_scheme else "",
                netloc,
                clean_path,
                parsed.params,
                sanitized_query,
                "",  # remove fragment hash which often contains tokens
            ))
            if not has_scheme:
                clean_url = clean_url.lstrip("/")
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
