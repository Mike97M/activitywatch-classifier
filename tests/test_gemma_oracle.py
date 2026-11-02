"""Unit tests for the GemmaOracle."""

from gemma_oracle import GemmaOracle


def test_gemma_heuristic_classification():
    oracle = GemmaOracle(backend="heuristic")

    # Coding signal
    res = oracle.classify_activity(app="foot", title="jdoe@laptop:~/Developement", url="")
    assert res["category"] == "Development"
    assert res["confidence"] >= 0.9

    # Social signal
    res = oracle.classify_activity(app="Brave-browser", title="Home / X", url="https://x.com/home")
    assert res["category"] == "Social Media"
    assert res["confidence"] >= 0.9

    # Communication signal
    res = oracle.classify_activity(app="Slack", title="Direct Messages - Alice", url="")
    assert res["category"] == "Communication"


def test_gemma_json_parser():
    oracle = GemmaOracle(backend="heuristic")
    cats = ["Development", "Communication", "Media & Entertainment"]

    valid_json = '```json\n{"category": "Development", "confidence": 0.98, "reasoning": "IDE active"}\n```'
    parsed = oracle._parse_llm_json(valid_json, cats)
    assert parsed is not None
    assert parsed["category"] == "Development"
    assert parsed["confidence"] == 0.98

    # Case-insensitive category match
    alt_json = '{"category": "communication", "confidence": 0.91, "reasoning": "chat client"}'
    parsed2 = oracle._parse_llm_json(alt_json, cats)
    assert parsed2 is not None
    assert parsed2["category"] == "Communication"


def test_gemma_local_model_info_and_fallback():
    oracle = GemmaOracle(backend="local")
    info = oracle.get_local_model_info()
    assert "available" in info
    assert "size_gb" in info

    # If no local model is currently downloaded, it should fall back safely without crashing
    res = oracle.classify_activity(app="Code", title="test.py - Visual Studio Code", url="")
    assert res["category"] == "Development"
    assert res["confidence"] >= 0.9
