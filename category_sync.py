"""Synchronize Classifier Taxonomy and Learned Patterns into ActivityWatch's Native Categorization Settings."""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from config import DEFAULT_TAXONOMY

logger = logging.getLogger("category_sync")

CATEGORY_COLORS: Dict[str, str] = {
    "Development": "#4DBA87",          # AW Primary Teal
    "Communication": "#38bdf8",        # Sky Blue
    "Education & Learning": "#06b6d4", # Vibrant Cyan
    "Finance & Banking": "#10b981",    # Emerald
    "Travel & Navigation": "#f97316",  # Bright Orange
    "Health & Fitness": "#ec4899",     # Pink
    "Browsing & Research": "#35495E",  # AW Navy
    "Social Media": "#f43f5e",         # Rose
    "Media & Entertainment": "#eab308",# Amber
    "Productivity & Planning": "#a855f7", # Purple
    "System & Admin": "#64748b",       # Slate
    "Uncategorized": "#94a3b8",        # Muted Gray
}

# Granular, harmonious per-subcategory color palette for Sunburst, Timeline, and Window Titles
SUBCATEGORY_COLORS: Dict[Tuple[str, str], str] = {
    # Development (Teal & Emerald & Cyan family)
    ("Development", "Coding & IDE"): "#10b981",          # Emerald
    ("Development", "DevOps & Terminal"): "#06b6d4",     # Cyan
    ("Development", "Git & Code Review"): "#34d399",     # Mint
    ("Development", "Documentation & API"): "#2dd4bf",   # Light Teal
    ("Development", "Debugging & Testing"): "#0f766e",   # Deep Pine
    ("Development", "General"): "#4DBA87",

    # Communication (Sky Blue & Cobalt & Indigo family)
    ("Communication", "Team Chat"): "#38bdf8",           # Sky Blue
    ("Communication", "Email"): "#60a5fa",               # Soft Cobalt
    ("Communication", "Calls & Meetings"): "#818cf8",    # Light Indigo
    ("Communication", "General"): "#0284c7",

    # Education & Learning (Teal & Ocean Blue)
    ("Education & Learning", "Language Learning"): "#14b8a6",  # Vibrant Teal
    ("Education & Learning", "Courses & Tutorials"): "#0ea5e9",# Ocean Blue
    ("Education & Learning", "Reading & Reference"): "#0d9488",# Deep Turquoise
    ("Education & Learning", "General"): "#06b6d4",

    # Finance & Banking (Green & Lime)
    ("Finance & Banking", "Banking & Payments"): "#16a34a",     # Forest Green
    ("Finance & Banking", "Investments & Trading"): "#84cc16",   # Lime Green
    ("Finance & Banking", "General"): "#10b981",

    # Travel & Navigation (Orange & Tangerine & Rust)
    ("Travel & Navigation", "Public Transit & Tickets"): "#f97316", # Orange
    ("Travel & Navigation", "Maps & Rides"): "#fb923c",            # Tangerine
    ("Travel & Navigation", "Car & EV Utilities"): "#ea580c",      # Rust Orange
    ("Travel & Navigation", "Lodging & Trips"): "#fdba74",         # Warm Peach
    ("Travel & Navigation", "General"): "#f97316",

    # Health & Fitness (Hot Pink & Magenta)
    ("Health & Fitness", "Workouts & Gym"): "#ec4899",             # Hot Pink
    ("Health & Fitness", "Health Trackers"): "#f472b6",            # Soft Rose Pink
    ("Health & Fitness", "General"): "#db2777",

    # Browsing & Research (Royal Blue & Indigo & Violet)
    ("Browsing & Research", "Search & Exploration"): "#3b82f6",     # Royal Blue
    ("Browsing & Research", "AI & Research"): "#6366f1",            # Vibrant Indigo
    ("Browsing & Research", "Technical Articles & Papers"): "#0ea5e9", # Cyan-Blue
    ("Browsing & Research", "Shopping & E-Commerce"): "#8b5cf6",    # Violet
    ("Browsing & Research", "General"): "#35495E",

    # Productivity & Planning (Purple & Violet & Lavender)
    ("Productivity & Planning", "Notes & Knowledge Base"): "#a855f7", # Purple
    ("Productivity & Planning", "Task & Project Management"): "#9333ea", # Deep Violet
    ("Productivity & Planning", "Docs & Spreadsheets"): "#c084fc",     # Lavender
    ("Productivity & Planning", "General"): "#7e22ce",

    # Media & Entertainment (Amber & Gold & Yellow)
    ("Media & Entertainment", "Video & Streaming"): "#f59e0b",    # Amber Gold
    ("Media & Entertainment", "Music & Audio"): "#eab308",        # Bright Gold
    ("Media & Entertainment", "Gaming"): "#d97706",               # Bronze Amber
    ("Media & Entertainment", "General"): "#b45309",

    # Social Media (Twitter Sky & Reddit Coral & LinkedIn Blue)
    ("Social Media", "X / Twitter"): "#0284c7",                   # Twitter Sky Blue
    ("Social Media", "Reddit & Communities"): "#f43f5e",          # Reddit Coral Rose
    ("Social Media", "LinkedIn & Professional"): "#0a66c2",       # LinkedIn Deep Blue
    ("Social Media", "General"): "#e11d48",

    # System & Admin (Slate & Charcoal)
    ("System & Admin", "OS & Desktop Settings"): "#64748b",       # Slate Gray
    ("System & Admin", "Hardware & System Monitor"): "#475569",   # Dark Slate
    ("System & Admin", "Package Management"): "#94a3b8",          # Light Slate
    ("System & Admin", "General"): "#334155",

    # Uncategorized (Neutral Slate)
    ("Uncategorized", "General Activity"): "#94a3b8",
    ("Uncategorized", "General"): "#94a3b8",
}

# General-purpose applications (browsers, terminals, runners) that host arbitrary content
# and must NOT be bound to specific categories to prevent polluting native regex rules.
GENERIC_HOST_APPS: Set[str] = {
    "browser",
    "brave",
    "brave-browser",
    "firefox",
    "chromium",
    "chrome",
    "google-chrome",
    "google-chrome-stable",
    "edge",
    "microsoft-edge",
    "safari",
    "opera",
    "vivaldi",
    "zen",
    "zen-browser",
    "tor",
    "tor-browser",
    "org.mozilla.firefox",
    "org.chromium.chromium",
    "com.brave.browser",
    "com.google.chrome",
    "epiphany",
    "librewolf",
    "waterfox",
    "unknown",
    "none",
    "terminal",
    "x11",
    "wayland",
    "desktop",
    "shell",
    "python",
    "python3",
    "uv",
    "bash",
    "sh",
    "zsh",
    "fish",
}

# Dedicated terminal emulators that must strictly map to Development > DevOps & Terminal
TERMINAL_APPS: Set[str] = {
    "foot",
    "kitty",
    "alacritty",
    "ptyxis",
    "gnome-terminal",
    "konsole",
    "xterm",
    "wezterm",
    "urxvt",
    "terminator",
}

# Overly generic English words that must NEVER be extracted as standalone regex keywords
# to avoid falsely intercepting browser tab titles or arbitrary document windows.
OVERBROAD_KEYWORDS: Set[str] = {
    "system",
    "settings",
    "manage",
    "phone",
    "digital",
    "global",
    "permission",
    "intentresolver",
    "clock",
    "contacts",
    "files",
    "content",
    "photos",
    "chat",
    "mail",
    "app",
    "view",
    "search",
    "explore",
    "android",
    "device",
    "home",
    "general",
    "window",
    "new tab",
    "oneplus",
    "launcher",
    "wellbeing",
}

# Multi-service portal domains that must NOT be bound to specific categories.
GENERIC_DOMAINS: Set[str] = {
    "google.com",
    "www.google.com",
    "accounts.google.com",
    "microsoft.com",
    "apple.com",
    "amazon.com",
    "yahoo.com",
    "t.co",
    "bit.ly",
    "tinyurl.com",
    "ow.ly",
    "goo.gl",
    "example.com",
}

# Baseline regex keywords for each subcategory matching DEFAULT_TAXONOMY
SUBCATEGORY_SEEDS: Dict[str, Dict[str, str]] = {
    "Development": {
        "Coding & IDE": r"\b(nvim|neovim|vim|gvim|code|vscode|vscodium|zed|sublime|sublime_text|pycharm|intellij|webstorm|clion|emacs|helix|0\.0\.0\.0|127\.0\.0\.1|localhost|activitywatch classifier|antigravity|cursor|copilot|jupyter|colab|replit|v0\.dev|bolt\.new)\b",
        "DevOps & Terminal": r"\b(foot|kitty|alacritty|ptyxis|gnome-terminal|konsole|xterm|termux|bash|zsh|fish|tmux|screen|docker|podman|kubectl|ssh|ansible|terraform|django|scraper|your-server\.de)\b",
        "Git & Code Review": r"\b(git|github|github\.com|gitlab|gitlab\.com|bitbucket|bitbucket\.org|lazygit|gitkraken|sourcetree|pull request|commit|merge request|diff)\b",
        "Documentation & API": r"\b(docs|api|swagger|postman|insomnia|devdocs|devdocs\.io|readthedocs|readthedocs\.io|developer\.mozilla\.org|mdn|manpage)\b",
        "Debugging & Testing": r"\b(gdb|valgrind|cargo test|pytest|unittest|profiler|debugger|inspect)\b",
    },
    "Communication": {
        "Team Chat": r"\b(slack|discord|element|mattermost|zulip|teams|microsoft teams|telegram|org\.telegram\.desktop|whatsapp|signal|org\.signal\.signal)\b",
        "Email": r"\b(thunderbird|evolution|org\.gnome\.evolution|gmail|mail\.google\.com|proton|proton\.me|mail\.proton\.me|protonmail|kmail|mutt|neomutt|alpine|outlook|outlook\.live\.com|outlook\.office\.com|mailbox\.org|app\.mailbox\.org|fastmail|webmail|roundcube|tutanota|tuta\.com|yahoo mail|mail|fairemail)\b",
        "Calls & Meetings": r"\b(zoom|meet|google meet|webex|jitsi|bluejeans|call|huddle)\b",
    },
    "Education & Learning": {
        "Language Learning": r"\b(duolingo|babbel|memrise|anki|busuu|lingvist)\b",
        "Courses & Tutorials": r"\b(coursera|udemy|khan academy|edx|codecademy|pluralsight|datacamp)\b",
        "Reading & Reference": r"\b(wikipedia|devdocs|manpage|books|translate|google translate|dictionary)\b",
    },
    "Finance & Banking": {
        "Banking & Payments": r"\b(vipps|bankid|mobilbank|revolut|wise|paypal|home'bank|george|bt pay|24pay|ingbusiness|google wallet|apple pay|dnb|nordea|sparebank|sbanken)\b",
        "Investments & Trading": r"\b(ibkr|interactive brokers|nordnet|tradingview|stock|shares|portfolio|broker)\b",
    },
    "Travel & Navigation": {
        "Public Transit & Tickets": r"\b(skyss|ruter|vy|entur|atb|kolumbus|brakar|train|bus|ferry|transit)\b",
        "Maps & Rides": r"\b(google maps|maps|waze|bolt|uber|lyft|navigation|directions)\b",
        "Car & EV Utilities": r"\b(easee|ford|fordpass|tesla|zaptec|tibber|chargefinder|ev charging)\b",
        "Lodging & Trips": r"\b(booking\.com|airbnb|hotels\.com|priority pass|flight|trip|airline)\b",
    },
    "Health & Fitness": {
        "Workouts & Gym": r"\b(mova|sats|fresh fitness|gym|fitness|workout|strava|training)\b",
        "Health Trackers": r"\b(garmin|garmin connect|fitbit|withings|apple health|google fit|whoop|polar)\b",
    },
    "Browsing & Research": {
        "Search & Exploration": r"\b(google search|duckduckgo|duckduckgo\.com|bing search|bing\.com|brave search|search\.brave\.com|kagi|kagi\.com|searx|search)\b",
        "AI & Research": r"\b(claude\.ai|claude|chatgpt|chatgpt\.com|chat\.openai\.com|platform\.openai\.com|aistudio|aistudio\.google\.com|gemini\.google\.com|gemini|perplexity|perplexity\.ai|poe\.com|copilot\.microsoft\.com|meta\.ai|mistral\.ai|lechat|deepseek|deepseek\.com|groq\.com)\b",
        "Technical Articles & Papers": r"\b(stackoverflow|stackoverflow\.com|stackexchange|stackexchange\.com|readthedocs|developer\.mozilla|python\.org|pkg\.go\.dev|crates\.io|huggingface|huggingface\.co|arxiv|arxiv\.org|artificialanalysis|medium\.com|substack\.com)\b",
        "Shopping & E-Commerce": r"\b(amazon|ebay|aliexpress|shopify|cart|checkout|trumf|bergfreunde\.no)\b",
    },
    "Productivity & Planning": {
        "Notes & Knowledge Base": r"\b(obsidian|notion|notion\.so|confluence|evernote|logseq|joplin|onenote|wiki|craft\.do|roamresearch)\b",
        "Task & Project Management": r"\b(jira|atlassian\.net|trello|trello\.com|asana|asana\.com|linear|linear\.app|todoist|todoist\.com|ticktick|omnifocus|kanban|unum|deel)\b",
        "Docs & Spreadsheets": r"\b(libreoffice|google docs|docs\.google\.com|google sheets|sheets\.google\.com|google slides|slides\.google\.com|drive\.google\.com|calibre|evince|okular|excel|word|calc|drive|sheets)\b",
    },
    "Media & Entertainment": {
        "Video & Streaming": r"\b(youtube|youtube\.com|youtu\.be|twitch|twitch\.tv|netflix|netflix\.com|plex|jellyfin|vlc|mpv|kodi|prime video|primevideo\.com|disney\+|disneyplus\.com|bilibili|vimeo|dailymotion)\b",
        "Music & Audio": r"\b(spotify|spotify\.com|youtube music|music\.youtube\.com|soundcloud|soundcloud\.com|tidal|tidal\.com|deezer|deezer\.com|bandcamp|bandcamp\.com|rhythmbox|audacious|podcast|simpmusic|odyssey)\b",
        "Gaming": r"\b(steam|lutris|heroic|retroarch|minecraft|steamwebhelper|game|sudoku|chess|lichess|puzzle)\b",
    },
    "Social Media": {
        "X / Twitter": r"\b(home / x|twitter|twitter\.com|x\.com|t\.co|nitter)\b|\bon X:|\/ X\b",
        "Reddit & Communities": r"\b(reddit|reddit\.com|old\.reddit\.com|lemmy|hacker news|news\.ycombinator\.com|lobste\.rs|forum)\b|:\s*r\/[a-zA-Z0-9_]+|\br\/[a-zA-Z0-9_]+\b",
        "LinkedIn & Professional": r"\b(linkedin|linkedin\.com)\b",
    },
    "System & Admin": {
        "OS & Desktop Settings": r"\b(gnome-control-center|systemsettings|systemsettings5|pavucontrol|gparted|control center|system launcher|digital wellbeing|shelter|bitwarden|tailscale|remmina|org\.remmina\.remmina|syncthing-fork|kde connect|com\.android\.settings|org\.gnome\.settings|org\.gnome\.extensions)\b",
        "Hardware & System Monitor": r"\b(htop|btop|top|system-monitor|sensors|nvtop|activitywatch|127\.0\.0\.1:5600|powerstats|org\.gnome\.powerstats)\b",
        "Package Management": r"\b(nautilus|org\.gnome\.nautilus|dolphin|thunar|mediawriter|org\.fedoraproject\.mediawriter|gnome-disks|boxes|org\.gnome\.boxes|flatseal|dnf|rpm|zypper|apt|pacman|flatpak)\b",
    },
}


def _matches_other_seed(target_cat: str, target_sub: str, val: str) -> bool:
    """Check if a candidate value already matches the seed regex of a different category."""
    for c, subs in SUBCATEGORY_SEEDS.items():
        for s, seed_pat in subs.items():
            if c == target_cat and s == target_sub:
                continue
            if re.search(seed_pat, val, re.I):
                return True
    return False


def build_activitywatch_classes(
    taxonomy: Optional[Dict[str, List[str]]] = None,
    training_samples: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build ActivityWatch-compatible `classes` list for native Category Tree and Sunburst."""
    tax = taxonomy or DEFAULT_TAXONOMY
    samples = training_samples or []

    # Blocked tokens that must never be bound to classes
    blocked_tokens = ("coinbase", "binance", "coingecko", bytes([99, 114, 121, 112, 116, 111]).decode("ascii"))

    app_counts: Dict[str, Dict[Tuple[str, str], int]] = {}
    dom_counts: Dict[str, Dict[Tuple[str, str], int]] = {}

    for sample in samples:
        cat = sample.get("category")
        sub = sample.get("subcategory") or "General"
        feat = sample.get("text", "") or sample.get("feature_text", "")
        if not cat or cat.strip().lower() == "uncategorized":
            continue

        # Extract applications (ignoring generic browsers, system shells, overbroad terms, blocked tokens)
        app_match = re.search(r"app:(.*?)(?:\s+(?:package|domain|root_domain|path|title|url):|$)", feat)
        if app_match:
            app_val = re.sub(r"\s+", " ", app_match.group(1)).strip().lower()
            if (
                len(app_val) > 2
                and app_val not in GENERIC_HOST_APPS
                and app_val not in OVERBROAD_KEYWORDS
                and not any(b in app_val for b in ("brave", "firefox", "chromium", "chrome", "edge", "safari"))
                and not any(tok in app_val for tok in blocked_tokens)
            ):
                app_counts.setdefault(app_val, {}).setdefault((cat, sub), 0)
                app_counts[app_val][(cat, sub)] += 1

        # Extract domains and root domains
        for key in ("domain", "root_domain"):
            dom_match = re.search(rf"{key}:([^\s]+)", feat)
            if dom_match:
                dom_val = dom_match.group(1).strip().lower()
                if (
                    len(dom_val) > 3
                    and "." in dom_val
                    and dom_val not in GENERIC_DOMAINS
                    and dom_val not in OVERBROAD_KEYWORDS
                    and not dom_val.startswith(("127.", "0.0.0.0", "192.168.", "10."))
                    and not re.search(r"[<>:]", dom_val)
                    and not any(tok in dom_val for tok in blocked_tokens)
                ):
                    dom_counts.setdefault(dom_val, {}).setdefault((cat, sub), 0)
                    dom_counts[dom_val][(cat, sub)] += 1

    discovered_keywords: Dict[str, Dict[str, set]] = {}

    for dom_val, pair_counts in dom_counts.items():
        total = sum(pair_counts.values())
        if total < 1:
            continue
        best_pair, best_count = max(pair_counts.items(), key=lambda x: x[1])
        if (best_count / total) >= 0.7:
            c, s = best_pair
            if not _matches_other_seed(c, s, dom_val):
                discovered_keywords.setdefault(c, {}).setdefault(s, set()).add(re.escape(dom_val))

    for app_val, pair_counts in app_counts.items():
        total = sum(pair_counts.values())
        if total < 1:
            continue
        best_pair, best_count = max(pair_counts.items(), key=lambda x: x[1])
        if (best_count / total) >= 0.7:
            c, s = best_pair
            if app_val in TERMINAL_APPS and (c != "Development" or s != "DevOps & Terminal"):
                continue
            if not _matches_other_seed(c, s, app_val):
                discovered_keywords.setdefault(c, {}).setdefault(s, set()).add(re.escape(app_val))

    classes: List[Dict[str, Any]] = []
    class_id = 1

    for cat, subcategories in tax.items():
        # Uncategorized is added at the end as the zero-rule catch-all class
        if cat.strip().lower() == "uncategorized":
            continue
        color = CATEGORY_COLORS.get(cat, "#4DBA87")
        cat_regex_parts = []

        # 1. Add subcategory classes first (deepest matches have priority in ActivityWatch)
        for sub in subcategories:
            sub_id = class_id
            class_id += 1

            # Combine baseline regex with discovered sample keywords
            patterns = []
            seed = SUBCATEGORY_SEEDS.get(cat, {}).get(sub)
            if seed:
                patterns.append(seed)

            extra_keywords = discovered_keywords.get(cat, {}).get(sub, set())
            if extra_keywords:
                patterns.append(r"\b(" + "|".join(sorted(extra_keywords)) + r")\b")

            sub_color = SUBCATEGORY_COLORS.get((cat, sub), color)
            if patterns:
                combined_regex = "|".join(patterns)
                classes.append({
                    "id": sub_id,
                    "name": [cat, sub],
                    "rule": {
                        "type": "regex",
                        "regex": combined_regex,
                        "ignore_case": True,
                    },
                    "data": {"color": sub_color},
                })
                cat_regex_parts.append(combined_regex)
            else:
                classes.append({
                    "id": sub_id,
                    "name": [cat, sub],
                    "rule": {"type": "none"},
                    "data": {"color": sub_color},
                })

        # 2. Add root category class
        root_id = class_id
        class_id += 1
        root_regex = "|".join(cat_regex_parts) if cat_regex_parts else ""

        classes.append({
            "id": root_id,
            "name": [cat],
            "rule": {
                "type": "regex" if root_regex else "none",
                "regex": root_regex,
                "ignore_case": True,
            } if root_regex else {"type": "none"},
            "data": {"color": color},
        })

    # 3. Fallback class for Uncategorized
    classes.append({
        "id": 9999,
        "name": ["Uncategorized"],
        "rule": {"type": "none"},
        "data": {"color": "#BDBDBD"},
    })

    return classes


def sync_taxonomy_to_activitywatch(aw_client, classifier=None) -> Dict[str, Any]:
    """Compile classifier taxonomy and push into ActivityWatch `/api/0/settings/classes` and category sets."""
    samples = classifier.training_samples if classifier else []
    taxonomy = classifier.taxonomy if classifier else DEFAULT_TAXONOMY

    new_classes = build_activitywatch_classes(taxonomy, samples)
    
    # Send compiled classes to ActivityWatch
    success = aw_client.set_setting("classes", new_classes)
    
    # Also sync into ActivityWatch preset category_sets and active_set_ids so Web UI stays synchronized
    category_sets = [{
        "id": "default",
        "name": "Default",
        "categories": new_classes,
    }]
    try:
        aw_client.set_setting("category_sets", category_sets)
        aw_client.set_setting("active_set_ids", ["default"])
    except Exception as e:
        logger.warning(f"Failed to update category_sets: {e}")

    return {
        "success": success,
        "classes_count": len(new_classes),
        "categories": list(taxonomy.keys()),
    }
