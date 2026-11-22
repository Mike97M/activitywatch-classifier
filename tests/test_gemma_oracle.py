"""Unit tests for the GemmaOracle."""

import math

from config import LLM_DEFAULT_CONFIDENCE
import pytest

from gemma_oracle import HEURISTIC_BACKEND, GemmaOracle, label_confidence_from_logprobs

# Nothing listens on the discard port: every call fails fast and exercises the heuristic fallback.
OFFLINE_URL = "http://127.0.0.1:9/v1"


def test_gemma_heuristic_classification():
    oracle = GemmaOracle(backend="openai_compatible", openai_url=OFFLINE_URL)

    # Coding signal
    res = oracle.classify_activity(app="foot", title="jdoe@laptop:~/Developement", url="")
    assert res["category"] == "Development"
    assert res["subcategory"] == "DevOps & Terminal"
    assert res["full_category"] == "Development > DevOps & Terminal"
    assert res["confidence"] >= 0.9

    # Social signal
    res = oracle.classify_activity(app="Brave-browser", title="Home / X", url="https://x.com/home")
    assert res["category"] == "Social Media"
    assert res["subcategory"] == "X / Twitter"
    assert res["confidence"] >= 0.9

    # Communication signal
    res = oracle.classify_activity(app="Slack", title="Direct Messages - Alice", url="")
    assert res["category"] == "Communication"
    assert res["subcategory"] == "Team Chat"


def test_gemma_json_parser():
    oracle = GemmaOracle(backend="openai_compatible", openai_url=OFFLINE_URL)
    tax = {
        "Development": ["Coding & IDE", "DevOps & Terminal"],
        "Communication": ["Team Chat", "Email"],
    }

    valid_json = '```json\n{"category": "Development", "subcategory": "Coding & IDE", "confidence": 0.98, "reasoning": "IDE active"}\n```'
    parsed = oracle._parse_llm_json(valid_json, tax)
    assert parsed is not None
    assert parsed["category"] == "Development"
    assert parsed["subcategory"] == "Coding & IDE"
    assert parsed["full_category"] == "Development > Coding & IDE"
    # Self-reported confidence is ignored (models just echo the prompt example)
    assert parsed["confidence"] == LLM_DEFAULT_CONFIDENCE

    # Case-insensitive category match
    alt_json = '{"category": "communication", "subcategory": "team chat", "confidence": 0.91, "reasoning": "chat client"}'
    parsed2 = oracle._parse_llm_json(alt_json, tax)
    assert parsed2 is not None
    assert parsed2["category"] == "Communication"
    assert parsed2["subcategory"] == "Team Chat"


def test_local_model_info_and_fallback_on_server_failure():
    oracle = GemmaOracle(backend="llama_server")
    info = oracle.get_local_model_info()
    assert "available" in info
    assert "size_gb" in info

    # If the LLM server is unreachable, results fall back to the heuristic and are tagged as such
    offline = GemmaOracle(backend="openai_compatible", openai_url=OFFLINE_URL)
    res = offline.classify_activity(app="Code", title="test.py - Visual Studio Code", url="")
    assert res["category"] == "Development"
    assert res["backend"] == HEURISTIC_BACKEND


def test_only_llama_server_and_openai_backends_are_accepted():
    for removed in ("local", "llama_cpp", "ollama", "gemini_api", "heuristic"):
        with pytest.raises(ValueError):
            GemmaOracle(backend=removed)
    oracle = GemmaOracle(backend="llama_server")
    with pytest.raises(ValueError):
        oracle.update_settings(backend="ollama", openai_url="http://changed/v1")
    assert oracle.backend == "llama_server"
    assert oracle.openai_url != "http://changed/v1"
    oracle.update_settings(backend="openai-compatible")
    assert oracle.backend == "openai_compatible"


def test_gemma_batch_classification():
    oracle = GemmaOracle(backend="openai_compatible", openai_url=OFFLINE_URL)
    activities = [
        {"app": "foot", "title": "user@laptop:~", "url": ""},
        {"app": "brave", "title": "Pull Request #123 · github.com", "url": "https://github.com/ActivityWatch/activitywatch/pull/123"},
        {"app": "thunderbird", "title": "Inbox (3 unread)", "url": ""},
    ]
    results = oracle.classify_batch(activities)
    assert len(results) == 3
    assert results[0]["category"] == "Development"
    assert results[0]["subcategory"] == "DevOps & Terminal"
    assert results[1]["category"] == "Development"
    assert results[1]["subcategory"] == "Git & Code Review"
    assert results[2]["category"] == "Communication"
    assert results[2]["subcategory"] == "Email"


def test_terminal_emulator_heuristics():
    oracle = GemmaOracle(backend="openai_compatible", openai_url=OFFLINE_URL)
    # Foot with user@laptop
    res = oracle.classify_activity(app="foot", title="<USER>@laptop:~", url="")
    assert res["category"] == "Development"
    assert res["subcategory"] == "DevOps & Terminal"

    # Terminal running btop should be hardware monitor
    res_btop = oracle.classify_activity(app="foot", title="btop", url="")
    assert res_btop["category"] == "System & Admin"
    assert res_btop["subcategory"] == "Hardware & System Monitor"


def test_label_confidence_from_logprobs_uses_only_label_values():
    tokens = [
        {"token": '{"category": "', "logprob": -5.0},  # scaffolding: ignored
        {"token": "Develop", "logprob": math.log(0.8)},
        {"token": "ment", "logprob": 0.0},
        {"token": '", "subcategory": "', "logprob": -5.0},
        {"token": "Coding & IDE", "logprob": math.log(0.5)},
        {"token": '"}', "logprob": -5.0},
    ]
    assert abs(label_confidence_from_logprobs(tokens) - 0.4) < 1e-9
    assert label_confidence_from_logprobs(None) is None
    assert label_confidence_from_logprobs([{"token": "garbage", "logprob": 0.0}]) is None


def test_grammar_has_no_confidence_field():
    oracle = GemmaOracle(backend="openai_compatible", openai_url=OFFLINE_URL)
    tax = {"Development": ["Coding & IDE"]}
    assert "confidence" not in oracle._build_grammar(tax)


def test_prompt_and_fallback_agree_with_rules_on_ai_and_terminal_tools():
    oracle = GemmaOracle(backend="openai_compatible", openai_url=OFFLINE_URL)
    prompt = oracle._build_prompt("firefox", "Claude", "https://claude.ai/chat/1", oracle.taxonomy)
    assert "'Browsing & Research > AI & Research'" in prompt
    assert "follow rule 2" in prompt
    res = oracle._smart_heuristic_classify("firefox", "ChatGPT", "https://chatgpt.com/c/1", oracle.taxonomy)
    assert res["full_category"] == "Browsing & Research > AI & Research"
    res = oracle._smart_heuristic_classify("firefox", "Notebook", "https://colab.research.google.com/x", oracle.taxonomy)
    assert res["category"] == "Development"
