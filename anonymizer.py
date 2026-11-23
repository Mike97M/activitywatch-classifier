"""PII & Sensitive Data Anonymizer for ActivityWatch window titles, URLs, and features.

Removes usernames, hostnames, private file paths, emails, phone numbers, bank/card numbers,
IP/MAC addresses, secrets/API keys, search queries and contact names before training samples
are saved. Training samples additionally get chat/email window titles masked (subjects and
contact names), see `sanitize_for_training`.

Add your own identifiers (real name, handles, employer, other device names) one per line in
data/private_terms.txt (git-ignored), or with ANONYMIZER_EXTRA_TERMS="Jane Doe,janedoe42,acme-laptop".
"""

import getpass
import ipaddress
import os
import re
import socket
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from config import DATA_DIR

PRIVATE_TERMS_FILE = DATA_DIR / "private_terms.txt"


def _configured_terms() -> List[str]:
    """ANONYMIZER_EXTRA_TERMS (comma-separated) plus data/private_terms.txt (one per line, # comments)."""
    terms = os.environ.get("ANONYMIZER_EXTRA_TERMS", "").split(",")
    try:
        terms += [ln.split("#", 1)[0] for ln in PRIVATE_TERMS_FILE.read_text(encoding="utf-8").splitlines()]
    except OSError:
        pass
    return terms

_GENERIC_NAMES = {"root", "user", "guest", "admin", "localhost", "home", "default"}

# Free-text search parameters: their values are what you typed into a search box.
_SEARCH_PARAMS = {
    "q", "query", "search", "search_query", "searchterm", "keyword", "keywords",
    "term", "text", "wd", "p", "k",
}

# Apps / web apps whose window titles are message subjects or contact/channel names.
_PRIVATE_TITLE_APPS = (
    "signal", "whatsapp", "telegram", "discord", "slack", "teams", "element", "thunderbird",
    "betterbird", "evolution", "geary", "mailspring", "kmail", "outlook", "messenger", "wire",
    "threema", "viber", "skype", "mutt",
)
_PRIVATE_TITLE_DOMAINS = (
    "mail.google.com", "outlook.live.com", "outlook.office.com", "outlook.office365.com",
    "mail.proton.me", "mailbox.org", "mail.yahoo.com", "fastmail.com", "mail.zoho.com",
    "tuta.com", "web.whatsapp.com", "web.telegram.org", "discord.com", "app.slack.com",
    "teams.microsoft.com", "teams.live.com", "messenger.com", "app.element.io",
)
# Words in those titles that still say *what kind* of activity it is (kept for the classifier).
_ACTIVITY_WORDS_RE = re.compile(
    r"(?i)\b(meeting|call|huddle|calendar|inbox|compose|drafts?|sent|spam|archive|settings)\b"
)

# Structured feature string built by FastClassifier.extract_features.
_FEATURE_RE = re.compile(
    r"^app:(?P<app>.*?) package:(?P<package>.*?) domain:(?P<domain>\S*) root_domain:\S*"
    r" path:.*? title:(?P<title>.*) url:\S*$",
    re.S,
)


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def _iban_ok(raw: str) -> bool:
    s = re.sub(r"\s", "", raw).upper()
    if not 15 <= len(s) <= 34:
        return False
    moved = s[4:] + s[:4]
    try:
        return int("".join(str(int(c, 36)) for c in moved)) % 97 == 1
    except ValueError:
        return False


def is_private_title_source(app: str, domain: str) -> bool:
    app, domain = (app or "").lower(), (domain or "").lower()
    return any(a in app for a in _PRIVATE_TITLE_APPS) or any(
        domain == d or domain.endswith("." + d) for d in _PRIVATE_TITLE_DOMAINS
    )


def _mask_private_title(title: str) -> str:
    words = sorted({w.lower() for w in _ACTIVITY_WORDS_RE.findall(title or "")})
    return " ".join(["<PRIVATE>", *words])


class TextAnonymizer:
    """Sanitizes text fields to ensure zero PII leaks into training datasets or Git."""

    def __init__(
        self,
        extra_usernames: Optional[List[str]] = None,
        extra_terms: Optional[Iterable[str]] = None,
        hostnames: Optional[Iterable[str]] = None,
    ):
        # --- Identities: login names (and their parts), hostnames, user-supplied terms ---
        logins: Set[str] = set()
        try:
            logins.add(getpass.getuser())
        except Exception:
            pass
        for env_var in ("USER", "LOGNAME", "USERNAME"):
            if os.environ.get(env_var):
                logins.add(os.environ[env_var])
        try:
            logins.add(Path.home().name)
        except Exception:
            pass
        logins.update(u for u in (extra_usernames or []) if u)

        self.usernames: Set[str] = set()
        for login in logins:
            login = login.lower()
            # "jdoe.guest" -> also "jdoe": the bare name shows up in titles and paths
            for part in [login, *re.split(r"[._-]", login)]:
                if len(part) >= 3 and part not in _GENERIC_NAMES:
                    self.usernames.add(part)

        if hostnames is None:
            try:
                hostnames = [socket.gethostname()]
            except Exception:
                hostnames = []
        self.hostnames: Set[str] = set()
        for host in hostnames:
            host = (host or "").lower()
            for name in (host, host.split(".")[0]):
                if len(name) >= 3 and name not in _GENERIC_NAMES:
                    self.hostnames.add(name)

        if extra_terms is None:
            extra_terms = _configured_terms()
        self.extra_terms: Set[str] = {t.strip().lower() for t in extra_terms if len(t.strip()) >= 3}

        def _alternation(terms: Set[str]) -> Optional[re.Pattern]:
            if not terms:
                return None
            # Longest first; the space in multi-word terms also matches "." "_" "-" "+" and URL-encoded "%20".
            # Letter-only boundaries, so handles glued to digits ("1497jdoe", "jdoe1497") are caught too.
            parts = [r"(?:[\s._+-]|%20)+".join(map(re.escape, t.split())) for t in sorted(terms, key=len, reverse=True)]
            return re.compile(rf"(?<![a-z]){'(?:' + '|'.join(parts) + ')'}(?![a-z])", re.IGNORECASE)

        self._user_re = _alternation(self.usernames | self.extra_terms)
        self._host_re = _alternation(self.hostnames)

        # --- Patterns (all case-insensitive: stored feature text is lowercased) ---
        self._email_re = re.compile(r"\b[A-Za-z0-9._%+-]+(?:@|%40)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")  # also URL-encoded "@"
        self._ipv4_re = re.compile(r"\b(?!(?:127\.0\.0\.1|0\.0\.0\.0)\b)(?:\d{1,3}\.){3}\d{1,3}\b")
        self._ipv6_candidate_re = re.compile(r"(?<![\w:])(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?![\w:])", re.I)
        self._mac_re = re.compile(r"\b(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}\b", re.I)
        self._phone_re = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
        # International: "+40 721 234 567", "+49 151 23456789" (needs a leading +, >= 8 digits)
        self._intl_phone_re = re.compile(r"(?<![\w+])\+\d{1,3}(?:[\s.-]?\(?\d{1,5}\)?){1,5}(?!\w)")
        self._card_re = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
        self._iban_re = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b", re.I)
        self._long_number_re = re.compile(r"\b\d{9,}\b")
        self._uuid_re = re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        )
        # Shell prompt "jdoe@laptop:~/src" -> "<USER>@<HOST>:~/src"
        self._shell_prompt_re = re.compile(r"\b([a-zA-Z0-9_.-]+)@([a-zA-Z0-9_.-]+):([~/\w.-]*)")
        self._home_path_re = re.compile(r"(?:/home|/Users)/[a-zA-Z0-9_.-]+(/[\w.-]+)*")
        self._win_path_re = re.compile(r"[A-Za-z]:\\Users\\[a-zA-Z0-9_.-]+(\\[\w.-]+)*", re.IGNORECASE)
        # "Direct Message with Sarah Connor", "Chat with Ștefan Popescu" (Unicode names)
        self._dm_re = re.compile(
            r"(?i)\b(direct messages?(?: with)?|dm with|chat with|conversation with|message to)"
            r"\s*[-:]?\s*([^\W\d_][\w.' ]{1,30})"
        )
        self._jwt_re = re.compile(
            r"\beyJ[a-zA-Z0-9_\-]{8,}\.[a-zA-Z0-9_\-]{8,}(?:\.[a-zA-Z0-9_\-]+)?\b", re.IGNORECASE
        )
        # Well-known secret formats, even without a "key=" in front of them
        self._secret_re = re.compile(
            r"\b(?:sk-(?:ant-|proj-|live-|test-)?[a-z0-9_-]{16,}|gh[pousr]_[a-z0-9]{20,}"
            r"|github_pat_[a-z0-9_]{20,}|glpat-[a-z0-9_-]{20,}|xox[abprs]-[a-z0-9-]{10,}"
            r"|akia[0-9a-z]{16}|aiza[0-9a-z_-]{35}|hf_[a-z0-9]{30,}|npm_[a-z0-9]{30,})",
            re.IGNORECASE,
        )
        # "OPENAI_API_KEY=...", "db_password:hunter2" (":" only without a space, so "Password: Settings" survives)
        self._secret_assign_re = re.compile(
            r"(?i)\b([a-z0-9_]*(?:token|secret|passw(?:or)?d|api[_-]?key|access[_-]?key|private[_-]?key)[a-z0-9_]*)"
            r"(?:\s*=\s*|:(?=\S))([\"']?)[^\s\"'&#]+"
        )
        self._hex_token_re = re.compile(r"\b[0-9a-fA-F]{24,64}\b")
        self._sensitive_kv_re = re.compile(
            r"(?i)\b("
            r"session(?:_id|_state)?|sid|token|access_token|id_token|refresh_token|"
            r"auth(?:user)?|secret|password|passwd|key|api_key|apikey|code|state|nonce|"
            r"csrf|xsrf|_csrf|phpsessid|jsessionid|aspsessionid|ticket|samlart|relaystate|"
            r"assertion|otp|totp|pin|passcode|credential|credentials|sig|signature"
            r")=([^&\s\"\'#]+)"
        )
        # Search-result page titles: "what I searched - Google Search"
        self._search_title_re = re.compile(
            r"(?i)^(.+?)(\s+[-–—]\s+(?:google search|google-suche|bing|brave search|startpage"
            r"|ecosia|qwant)\b|\s+at duckduckgo\b)"
        )
        self._sensitive_param_names = {
            "token", "key", "auth", "api_key", "apikey", "code", "access_token",
            "refresh_token", "secret", "password", "passwd", "session", "session_id",
            "sessionid", "session_state", "sid", "sig", "signature", "jwt", "id_token",
            "credential", "credentials", "state", "nonce", "csrf", "xsrf", "_csrf",
            "ticket", "authuser", "client_id", "client_secret", "phpsessid", "jsessionid",
            "aspsessionid", "samlart", "relaystate", "assertion", "otp", "totp", "pin",
            "passcode", "user_id", "userid", "account_id", "email", "phone", "username", "login",
            "login_hint", "user", "registered",
        }

    # ------------------------------------------------------------------
    def _mask_ipv6(self, m: re.Match) -> str:
        try:
            ip = ipaddress.IPv6Address(m.group(0))
        except ValueError:
            return m.group(0)  # times like 12:30:45, not an address
        return m.group(0) if ip.is_loopback or ip.is_unspecified else "<IP>"

    def _mask_card(self, m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return "<CARD>" if 13 <= len(digits) <= 19 and _luhn_ok(digits) else m.group(0)

    def _mask_intl_phone(self, m: re.Match) -> str:
        return "<PHONE>" if len(re.sub(r"\D", "", m.group(0))) >= 8 else m.group(0)

    def sanitize_title(self, title: str) -> str:
        """Sanitize window title removing paths, usernames, prompt info, emails, tokens, and PII."""
        if not title:
            return ""
        text = title.strip()

        # Secrets first, while their surrounding structure is intact
        text = self._jwt_re.sub("<JWT>", text)
        text = self._secret_re.sub("<SECRET>", text)
        text = self._secret_assign_re.sub(r"\1=<REDACTED>", text)
        text = self._sensitive_kv_re.sub(r"\1=<REDACTED>", text)
        text = self._hex_token_re.sub("<TOKEN>", text)

        text = self._search_title_re.sub(r"<QUERY>\2", text)
        text = self._shell_prompt_re.sub(r"<USER>@<HOST>:\3", text)
        text = self._home_path_re.sub("~/<PATH>", text)
        text = self._win_path_re.sub(r"C:\\Users\\<USER>\\<PATH>", text)
        text = self._email_re.sub("<EMAIL>", text)

        # Numbers: validated formats before the generic ones
        text = self._iban_re.sub(lambda m: "<IBAN>" if _iban_ok(m.group(0)) else m.group(0), text)
        text = self._card_re.sub(self._mask_card, text)
        text = self._intl_phone_re.sub(self._mask_intl_phone, text)
        text = self._mac_re.sub("<MAC>", text)
        text = self._ipv6_candidate_re.sub(self._mask_ipv6, text)
        text = self._ipv4_re.sub("<IP>", text)
        text = self._uuid_re.sub("<UUID>", text)
        text = self._long_number_re.sub("<NUMBER>", text)
        text = self._phone_re.sub("<PHONE>", text)

        text = self._dm_re.sub(r"\1 - <CONTACT>", text)
        if self._host_re:
            text = self._host_re.sub("<HOST>", text)
        if self._user_re:
            text = self._user_re.sub("<USER>", text)
        return text

    def sanitize_text(self, text: str) -> str:
        """Sanitize a feature string or mixed text; chat/email titles inside feature strings are masked."""
        if not text:
            return ""
        m = _FEATURE_RE.match(text)
        if m and is_private_title_source(m.group("app") + " " + m.group("package"), m.group("domain")):
            a, b = m.span("title")
            text = text[:a] + _mask_private_title(m.group("title")) + text[b:]
        return self.sanitize_title(text)

    def sanitize_url(self, url: str) -> str:
        """Sanitize URLs by stripping sensitive query params, tokens, path hashes, and basic auth credentials."""
        if not url:
            return ""
        try:
            raw_url = url.strip()
            has_scheme = "://" in raw_url
            parsed = urlparse(raw_url if has_scheme else f"https://{raw_url}")
            if not parsed.netloc:
                return self.sanitize_title(url)

            netloc = parsed.netloc.split("@")[-1]  # drop basic auth user:pass@
            netloc = self._ipv4_re.sub("<IP>", netloc)

            clean_segments = []
            for seg in parsed.path.split("/"):
                if self._hex_token_re.fullmatch(seg):
                    seg = "<TOKEN>"
                elif self._uuid_re.fullmatch(seg):
                    seg = "<UUID>"
                else:
                    seg = self._uuid_re.sub("<UUID>", seg)
                    seg = self._hex_token_re.sub("<TOKEN>", seg)
                    seg = self._email_re.sub("<EMAIL>", seg)
                    seg = self._secret_re.sub("<SECRET>", seg)
                    seg = self._long_number_re.sub("<NUMBER>", seg)
                clean_segments.append(seg)
            clean_path = "/".join(clean_segments)
            if self._host_re:
                clean_path = self._host_re.sub("<HOST>", clean_path)
            if self._user_re:
                clean_path = self._user_re.sub("<USER>", clean_path)

            clean_pairs = []
            for k, v in parse_qsl(parsed.query, keep_blank_values=True):
                k_lower = k.lower()
                if k_lower in _SEARCH_PARAMS and v:
                    clean_pairs.append((k, "<QUERY>"))
                elif (
                    k_lower in self._sensitive_param_names
                    or len(v) > 24
                    or self._hex_token_re.search(v)
                    or self._jwt_re.search(v)
                    or self._secret_re.search(v)
                ):
                    clean_pairs.append((k, "<REDACTED>"))
                elif self._email_re.search(v):
                    clean_pairs.append((k, "<EMAIL>"))
                elif self._uuid_re.search(v):
                    clean_pairs.append((k, "<UUID>"))
                else:
                    clean_pairs.append((k, v))

            clean_url = urlunparse((
                parsed.scheme if has_scheme else "",
                netloc,
                clean_path,
                parsed.params,
                urlencode(clean_pairs),
                "",  # fragments often carry tokens
            ))
            return clean_url if has_scheme else clean_url.lstrip("/")
        except Exception:
            return self.sanitize_title(url)

    def sanitize_event_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean event payload free of PII (used for everything, including the ActivityWatch output)."""
        cleaned = dict(data)
        if "title" in cleaned:
            cleaned["title"] = self.sanitize_title(str(cleaned["title"]))
        if "url" in cleaned:
            cleaned["url"] = self.sanitize_url(str(cleaned["url"]))
        return cleaned

    def sanitize_for_training(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Stricter cleaning for stored training samples: chat/email titles (contacts, subjects) are masked.

        Your ActivityWatch output keeps the real titles; only what is learned and saved is masked.
        """
        cleaned = self.sanitize_event_data(data)
        url = str(cleaned.get("url") or "")
        domain = urlparse(url if "://" in url else f"https://{url}").netloc.split(":")[0] if url else ""
        app = f"{cleaned.get('app') or ''} {cleaned.get('package') or ''}"
        if cleaned.get("title") and is_private_title_source(app, domain):
            cleaned["title"] = _mask_private_title(str(cleaned["title"]))
        return cleaned


# Global singleton instance
anonymizer = TextAnonymizer()
