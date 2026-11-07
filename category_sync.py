"""Synchronize Classifier Taxonomy and Learned Patterns into ActivityWatch's Native Categorization Settings."""

import logging
import re
from typing import Any, Dict, List, Optional, Set
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

# Multi-service portal domains that must NOT be bound to specific categories.
GENERIC_DOMAINS: Set[str] = {
    "google.com",
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
        "Coding & IDE": r"\b(nvim|neovim|vim|gvim|code|vscode|vscodium|zed|sublime|sublime_text|pycharm|intellij|webstorm|clion|emacs|helix|0\.0\.0\.0|localhost|activitywatch classifier|claude\.ai|claude|chatgpt|aistudio|antigravity)\b",
        "DevOps & Terminal": r"\b(foot|kitty|alacritty|ptyxis|gnome-terminal|konsole|xterm|termux|bash|zsh|fish|tmux|screen|docker|podman|kubectl|ssh|ansible|terraform|django|scraper|your-server\.de)\b",
        "Git & Code Review": r"\b(git|github|gitlab|bitbucket|lazygit|gitkraken|sourcetree|pull request|commit)\b",
        "Documentation & API": r"\b(docs|api|swagger|postman|insomnia|devdocs|manpage)\b",
        "Debugging & Testing": r"\b(gdb|valgrind|cargo test|pytest|unittest|profiler|debugger|inspect)\b",
    },
    "Communication": {
        "Team Chat": r"\b(slack|discord|element|mattermost|zulip|teams|microsoft teams|telegram|org\.telegram\.desktop|whatsapp|signal|org\.signal\.signal)\b",
        "Email": r"\b(thunderbird|evolution|org\.gnome\.evolution|gmail|proton|protonmail|kmail|mutt|neomutt|alpine|outlook|mailbox\.org|app\.mailbox\.org|fastmail|webmail|roundcube|tutanota|yahoo mail|mail|fairemail)\b",
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
        "Search & Exploration": r"\b(google search|duckduckgo|bing search|brave search|kagi|searx|search)\b",
        "Technical Articles & Papers": r"\b(stackoverflow|stackexchange|github\.com|gitlab\.com|readthedocs|developer\.mozilla|python\.org|pkg\.go\.dev|crates\.io|huggingface|arxiv|artificialanalysis)\b",
        "Shopping & E-Commerce": r"\b(amazon|ebay|aliexpress|shopify|cart|checkout|trumf|bergfreunde\.no)\b",
    },
    "Productivity & Planning": {
        "Notes & Knowledge Base": r"\b(obsidian|notion|confluence|evernote|logseq|joplin|onenote|wiki)\b",
        "Task & Project Management": r"\b(jira|trello|asana|linear|todoist|ticktick|omnifocus|kanban|unum|deel)\b",
        "Docs & Spreadsheets": r"\b(libreoffice|google docs|google sheets|calibre|evince|okular|excel|word|calc|drive|sheets)\b",
    },
    "Media & Entertainment": {
        "Video & Streaming": r"\b(youtube|twitch|netflix|plex|jellyfin|vlc|mpv|kodi|prime video|disney\+|bilibili)\b",
        "Music & Audio": r"\b(spotify|youtube music|soundcloud|tidal|deezer|rhythmbox|audacious|podcast|simpmusic|odyssey)\b",
        "Gaming": r"\b(steam|lutris|heroic|retroarch|minecraft|steamwebhelper|game|sudoku|chess|lichess|puzzle)\b",
    },
    "Social Media": {
        "X / Twitter": r"\b(home / x|twitter|x\.com)\b",
        "Reddit & Communities": r"\b(reddit|lemmy|hacker news|lobste\.rs|forum)\b",
        "LinkedIn & Professional": r"\b(linkedin)\b",
    },
    "System & Admin": {
        "OS & Desktop Settings": r"\b(gnome-control-center|systemsettings|pavucontrol|gparted|settings|control center|launcher|system launcher|clock|wellbeing|shelter|bitwarden|tailscale|remmina|org\.remmina\.remmina|syncthing-fork|kde connect)\b",
        "Hardware & System Monitor": r"\b(htop|btop|top|system-monitor|sensors|nvtop|activitywatch|127\.0\.0\.1:5600|powerstats|org\.gnome\.powerstats)\b",
        "Package Management": r"\b(nautilus|org\.gnome\.nautilus|dolphin|thunar|mediawriter|org\.fedoraproject\.mediawriter|gnome-disks|boxes|org\.gnome\.boxes|flatseal|dnf|rpm|zypper|apt|pacman|flatpak)\b",
    },
}


def build_activitywatch_classes(
    taxonomy: Optional[Dict[str, List[str]]] = None,
    training_samples: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build ActivityWatch-compatible `classes` list for native Category Tree and Sunburst."""
    tax = taxonomy or DEFAULT_TAXONOMY
    samples = training_samples or []

    # Aggregate discovered keywords per (category, subcategory) from training samples
    # Uncategorized is strictly a fallback catch-all and must NEVER learn or bind keywords.
    discovered_keywords: Dict[str, Dict[str, set]] = {}
    for sample in samples:
        cat = sample.get("category")
        sub = sample.get("subcategory") or "General"
        feat = sample.get("text", "") or sample.get("feature_text", "")
        if not cat or cat.strip().lower() == "uncategorized":
            continue
        discovered_keywords.setdefault(cat, {}).setdefault(sub, set())

        # Extract applications (ignoring generic browsers, system shells, and enforcing terminal boundaries)
        app_match = re.search(r"app:([^\s]+)", feat)
        if app_match:
            app_val = app_match.group(1).strip().lower()
            if (
                len(app_val) > 2
                and app_val not in GENERIC_HOST_APPS
                and not any(b in app_val for b in ("brave", "firefox", "chromium", "chrome", "edge", "safari"))
            ):
                # Ensure terminal apps can only belong to Development > DevOps & Terminal
                if app_val in TERMINAL_APPS:
                    if cat == "Development" and sub == "DevOps & Terminal":
                        discovered_keywords[cat][sub].add(re.escape(app_val))
                else:
                    discovered_keywords[cat][sub].add(re.escape(app_val))

        # Extract domains (ignoring multi-service roots like google.com and local endpoints)
        domain_match = re.search(r"domain:([^\s]+)", feat)
        if domain_match:
            dom_val = domain_match.group(1).strip().lower()
            if (
                len(dom_val) > 3
                and "." in dom_val
                and dom_val not in GENERIC_DOMAINS
                and not dom_val.startswith(("127.", "0.0.0.0", "192.168.", "10."))
                and not re.search(r"[<>:]", dom_val)
            ):
                discovered_keywords[cat][sub].add(re.escape(dom_val))

        # Extract root domains (only if not a multi-service portal domain like google.com)
        root_match = re.search(r"root_domain:([^\s]+)", feat)
        if root_match:
            root_val = root_match.group(1).strip().lower()
            if (
                len(root_val) > 3
                and "." in root_val
                and root_val not in GENERIC_DOMAINS
                and not root_val.startswith(("127.", "0.0.0.0", "192.168.", "10."))
                and not re.search(r"[<>:]", root_val)
            ):
                discovered_keywords[cat][sub].add(re.escape(root_val))

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
                    "data": {"color": color},
                })
                cat_regex_parts.append(combined_regex)
            else:
                classes.append({
                    "id": sub_id,
                    "name": [cat, sub],
                    "rule": {"type": "none"},
                    "data": {"color": color},
                })

        # 2. Add root category class
        root_id = class_id
        class_id += 1
        root_regex = "|".join(cat_regex_parts) if cat_regex_parts else ""

        # For Browsing & Research, add generic browser app regex at the root level (depth 1).
        # In ActivityWatch, deeper child classes (depth 2, e.g. YouTube, GitHub, Duolingo, Gmail)
        # take precedence over parent classes. So specific URL rules win, and only general browsing
        # falls back to the parent Browsing & Research class.
        if cat == "Browsing & Research":
            browser_fallback = r"\b(firefox|chrome|google-chrome|brave|brave-browser|chromium|microsoft-edge|edge|opera|vivaldi|zen|zen-browser|safari|epiphany|browser)\b"
            root_regex = f"{browser_fallback}|{root_regex}" if root_regex else browser_fallback

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
