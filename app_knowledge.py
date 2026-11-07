"""Application Knowledge Base and Automated Web Search Resolver for Mobile & Desktop Apps.

Provides curated domain intelligence and on-demand web search resolution for mobile (e.g. Android)
and desktop applications with sparse window titles (e.g. Skyss, Vipps, Duolingo, Easee, MOVA).
"""

import html
import json
import logging
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from config import DATA_DIR

logger = logging.getLogger("app_knowledge")
APP_KNOWLEDGE_FILE = DATA_DIR / "app_knowledge.json"

# Pre-seeded high-confidence knowledge for mobile & desktop apps
PRESEEDED_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    # Education & Learning
    "duolingo": {
        "description": "Language learning educational app with interactive lessons and vocabulary practice.",
        "category": "Education & Learning",
        "subcategory": "Language Learning",
    },
    "babbel": {
        "description": "Language learning subscription app for conversation and grammar training.",
        "category": "Education & Learning",
        "subcategory": "Language Learning",
    },
    "memrise": {
        "description": "Language learning flashcard and spaced-repetition study app.",
        "category": "Education & Learning",
        "subcategory": "Language Learning",
    },
    "anki": {
        "description": "Spaced repetition flashcard software for learning and memorization.",
        "category": "Education & Learning",
        "subcategory": "Language Learning",
    },
    "translate": {
        "description": "Language translation dictionary and text translation tool.",
        "category": "Education & Learning",
        "subcategory": "Reading & Reference",
    },
    "google translate": {
        "description": "Multi-language neural machine translation service.",
        "category": "Education & Learning",
        "subcategory": "Reading & Reference",
    },
    # Travel & Navigation / Mobility
    "skyss": {
        "description": "Official public transport ticket and travel route planning app for Vestland / Bergen, Norway.",
        "category": "Travel & Navigation",
        "subcategory": "Public Transit & Tickets",
    },
    "ruter": {
        "description": "Public transportation travel planner and mobile ticket purchase app in Oslo, Norway.",
        "category": "Travel & Navigation",
        "subcategory": "Public Transit & Tickets",
    },
    "vy": {
        "description": "Norwegian national railway train and bus ticket booking app.",
        "category": "Travel & Navigation",
        "subcategory": "Public Transit & Tickets",
    },
    "entur": {
        "description": "National journey planner for public transit across all of Norway.",
        "category": "Travel & Navigation",
        "subcategory": "Public Transit & Tickets",
    },
    "maps": {
        "description": "Digital map navigation, GPS route directions, and local business discovery.",
        "category": "Travel & Navigation",
        "subcategory": "Maps & Rides",
    },
    "google maps": {
        "description": "GPS navigation, satellite mapping, live traffic, and route directions.",
        "category": "Travel & Navigation",
        "subcategory": "Maps & Rides",
    },
    "waze": {
        "description": "Community-driven GPS navigation with live traffic alerts and police warnings.",
        "category": "Travel & Navigation",
        "subcategory": "Maps & Rides",
    },
    "bolt": {
        "description": "On-demand ride-hailing taxi and electric scooter sharing service.",
        "category": "Travel & Navigation",
        "subcategory": "Maps & Rides",
    },
    "uber": {
        "description": "Ride-sharing, taxi dispatch, and personal urban transportation.",
        "category": "Travel & Navigation",
        "subcategory": "Maps & Rides",
    },
    "booking.com": {
        "description": "Online hotel lodging, flight reservation, and travel booking service.",
        "category": "Travel & Navigation",
        "subcategory": "Lodging & Trips",
    },
    "priority pass": {
        "description": "Global airport lounge access and VIP traveler membership service.",
        "category": "Travel & Navigation",
        "subcategory": "Lodging & Trips",
    },
    "easee": {
        "description": "Smart electric vehicle (EV) charging box management, energy monitoring, and remote control app.",
        "category": "Travel & Navigation",
        "subcategory": "Car & EV Utilities",
    },
    "ford": {
        "description": "FordPass connected vehicle management, remote engine start, lock/unlock, and car telematics.",
        "category": "Travel & Navigation",
        "subcategory": "Car & EV Utilities",
    },
    # Finance & Banking
    "vipps": {
        "description": "Norwegian mobile payment, peer-to-peer money transfer, bill pay, and instant checkout app.",
        "category": "Finance & Banking",
        "subcategory": "Banking & Payments",
    },
    "bankid": {
        "description": "Official Norwegian digital identity and secure banking authentication service.",
        "category": "Finance & Banking",
        "subcategory": "Banking & Payments",
    },
    "mobilbank": {
        "description": "Mobile banking app for checking account balance, paying bills, and managing personal bank accounts.",
        "category": "Finance & Banking",
        "subcategory": "Banking & Payments",
    },
    "revolut": {
        "description": "Digital banking, international currency exchange, debit card, and budgeting app.",
        "category": "Finance & Banking",
        "subcategory": "Banking & Payments",
    },
    "wise": {
        "description": "Multi-currency bank account and low-fee international money transfers.",
        "category": "Finance & Banking",
        "subcategory": "Banking & Payments",
    },
    "paypal": {
        "description": "Digital payment system, online money transfers, and electronic wallet.",
        "category": "Finance & Banking",
        "subcategory": "Banking & Payments",
    },
    "home'bank": {
        "description": "ING mobile banking application for personal financial accounts and payments.",
        "category": "Finance & Banking",
        "subcategory": "Banking & Payments",
    },
    "ingbusiness": {
        "description": "ING commercial banking app for business banking and corporate payments.",
        "category": "Finance & Banking",
        "subcategory": "Banking & Payments",
    },
    "george": {
        "description": "Erste Group George mobile banking application for accounts and money transfers.",
        "category": "Finance & Banking",
        "subcategory": "Banking & Payments",
    },
    "bt pay": {
        "description": "Banca Transilvania contactless mobile wallet and card management app.",
        "category": "Finance & Banking",
        "subcategory": "Banking & Payments",
    },
    "24pay": {
        "description": "Mobile digital wallet and public services / bills contactless payment app.",
        "category": "Finance & Banking",
        "subcategory": "Banking & Payments",
    },
    "google wallet": {
        "description": "Digital contactless mobile wallet for credit cards, passes, and transit tickets.",
        "category": "Finance & Banking",
        "subcategory": "Banking & Payments",
    },
    "ibkr mobile": {
        "description": "Interactive Brokers mobile trading platform for stocks, options, and global markets.",
        "category": "Finance & Banking",
        "subcategory": "Investments & Trading",
    },
    # Health & Fitness
    "mova": {
        "description": "Norwegian fitness center and gym training membership app for workouts and classes.",
        "category": "Health & Fitness",
        "subcategory": "Workouts & Gym",
    },
    "garmin connect": {
        "description": "Fitness tracking, health telemetry, GPS workout analysis, and Garmin smartwatch syncing.",
        "category": "Health & Fitness",
        "subcategory": "Health Trackers",
    },
    # Media & Entertainment
    "simpmusic": {
        "description": "Open-source YouTube Music player client for streaming songs, playlists, and audio.",
        "category": "Media & Entertainment",
        "subcategory": "Music & Audio",
    },
    "odyssey": {
        "description": "Audiobook and offline audio player application.",
        "category": "Media & Entertainment",
        "subcategory": "Music & Audio",
    },
    "sudoku": {
        "description": "Classic logic-based numbers placement puzzle game.",
        "category": "Media & Entertainment",
        "subcategory": "Gaming",
    },
    "lichess": {
        "description": "Free and open-source chess server, puzzle trainer, and online chess games.",
        "category": "Media & Entertainment",
        "subcategory": "Gaming",
    },
    # Development
    "termux": {
        "description": "Android terminal emulator and Linux environment running shell packages and command-line tools.",
        "category": "Development",
        "subcategory": "DevOps & Terminal",
    },
    "claude": {
        "description": "Anthropic AI assistant app for programming, reasoning, and technical analysis.",
        "category": "Development",
        "subcategory": "Coding & IDE",
    },
    "antigravity": {
        "description": "AI-assisted development agent and IDE coding environment.",
        "category": "Development",
        "subcategory": "Coding & IDE",
    },
    # Productivity & Planning
    "unum": {
        "description": "Visual content planning, grid design, and scheduling tool for creative social accounts.",
        "category": "Productivity & Planning",
        "subcategory": "Task & Project Management",
    },
    "deel": {
        "description": "Global contractor payroll, compliance, and international team payment platform.",
        "category": "Productivity & Planning",
        "subcategory": "Task & Project Management",
    },
    "trumf": {
        "description": "Norwegian grocery customer loyalty rewards, cashback bonus, and digital coupon program.",
        "category": "Browsing & Research",
        "subcategory": "Shopping & E-Commerce",
    },
    "bergfreunde.no": {
        "description": "Online retailer and shop for outdoor apparel, hiking gear, and climbing equipment in Norway.",
        "category": "Browsing & Research",
        "subcategory": "Shopping & E-Commerce",
    },
    # System & Admin
    "shelter": {
        "description": "Free and open-source Android sandbox utilizing the Work Profile to isolate apps and privacy.",
        "category": "System & Admin",
        "subcategory": "OS & Desktop Settings",
    },
    "bitwarden": {
        "description": "Open-source password manager and secure credential vault.",
        "category": "System & Admin",
        "subcategory": "OS & Desktop Settings",
    },
    "tailscale": {
        "description": "Zero-config mesh VPN service connecting private devices securely over WireGuard.",
        "category": "System & Admin",
        "subcategory": "OS & Desktop Settings",
    },
    "syncthing-fork": {
        "description": "Decentralized, encrypted peer-to-peer continuous file synchronization utility.",
        "category": "System & Admin",
        "subcategory": "OS & Desktop Settings",
    },
    "kde connect": {
        "description": "Multi-device wireless companion tool for notifications, file transfers, and shared clipboard.",
        "category": "System & Admin",
        "subcategory": "OS & Desktop Settings",
    },
    "digital wellbeing": {
        "description": "Android screen time management, daily app timer limits, and device usage dashboard.",
        "category": "System & Admin",
        "subcategory": "OS & Desktop Settings",
    },
}


class AppKnowledgeResolver:
    """Manages persistent application knowledge and triggers web search for unknown apps."""

    def __init__(self, storage_path: Path = APP_KNOWLEDGE_FILE):
        self.storage_path = storage_path
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        """Load stored knowledge and merge with preseeded definitions."""
        self._cache = dict(PRESEEDED_KNOWLEDGE)
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._cache.update(data)
            except Exception as e:
                logger.warning(f"Failed to read app knowledge file {self.storage_path}: {e}")

    def _save(self):
        """Save cached knowledge persistently."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to write app knowledge file {self.storage_path}: {e}")

    def normalize_app_name(self, name: str) -> str:
        """Clean and normalize app string."""
        if not name:
            return ""
        clean = name.strip().lower()
        # Remove non-breaking spaces and redundant prefixes
        clean = clean.replace("\xa0", " ").strip()
        return clean

    def lookup(self, app_name: str, enable_search: bool = True) -> Optional[Dict[str, Any]]:
        """Lookup app knowledge, querying web search if missing and enable_search is True."""
        key = self.normalize_app_name(app_name)
        if not key:
            return None

        # 1. Exact match
        if key in self._cache:
            return self._cache[key]

        # 2. Check partial word matches in cache
        for k, val in self._cache.items():
            if k == key or (len(key) > 3 and key in k) or (len(k) > 3 and k in key):
                return val

        # 3. Web Search resolution if enabled
        if enable_search:
            info = self.search_web_for_app(app_name)
            if info:
                self._cache[key] = info
                self._save()
                return info

        return None

    def search_web_for_app(self, app_name: str) -> Optional[Dict[str, Any]]:
        """Search DuckDuckGo Lite / Wikipedia to determine the purpose of an unknown app."""
        clean_name = self.normalize_app_name(app_name)
        if len(clean_name) < 2:
            return None

        query = f"{clean_name} android app"
        logger.info(f"Querying web search for unknown app '{app_name}'...")

        try:
            # First try DuckDuckGo Lite GET
            url = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote(query)}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                text = resp.read().decode("utf-8", errors="ignore")
                tds = re.findall(r"<td[^>]*>(.*?)</td>", text, re.DOTALL)
                snippets = []
                for t in tds:
                    t_clean = re.sub(r"<[^>]+>", " ", html.unescape(t)).strip()
                    t_clean = " ".join(t_clean.split())
                    if (
                        len(t_clean) > 30
                        and "DuckDuckGo" not in t_clean
                        and "Next >" not in t_clean
                        and "Zero-click" not in t_clean
                    ):
                        snippets.append(t_clean)
                        if len(snippets) >= 2:
                            break

                if snippets:
                    summary = " ".join(snippets)[:250]
                    logger.info(f"Discovered info for '{app_name}': {summary[:80]}...")
                    return {
                        "description": summary,
                        "discovered_via": "web_search",
                    }
        except Exception as e:
            logger.debug(f"Search lookup failed for '{app_name}': {e}")

        # Secondary: Wikipedia search
        try:
            wiki_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(app_name)}&limit=1&format=json"
            req = urllib.request.Request(wiki_url, headers={"User-Agent": "aw-classifier/1.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode())
                if len(data) > 2 and data[2] and len(data[2][0]) > 20:
                    summary = data[2][0][:250]
                    return {
                        "description": summary,
                        "discovered_via": "wikipedia",
                    }
        except Exception:
            pass

        return None

    def get_description(self, app_name: str) -> Optional[str]:
        """Return 1-sentence descriptive summary of app."""
        info = self.lookup(app_name, enable_search=False)
        return info.get("description") if info else None

    def get_category_hint(self, app_name: str) -> Optional[Tuple[str, str]]:
        """Return (Category, Subcategory) hint if recognized in knowledge base."""
        info = self.lookup(app_name, enable_search=False)
        if info and "category" in info and "subcategory" in info:
            return info["category"], info["subcategory"]
        return None


# Global singleton instance
app_knowledge = AppKnowledgeResolver()
