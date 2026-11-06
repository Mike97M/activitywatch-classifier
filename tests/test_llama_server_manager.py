"""Unit tests for LlamaServerManager."""

from pathlib import Path
from unittest.mock import patch, MagicMock
from llama_server_manager import (
    LlamaServerManager,
    get_llama_server_binary,
    download_llama_server,
)


def test_get_llama_server_binary():
    bin_path = get_llama_server_binary()
    assert bin_path is not None
    assert bin_path.name == "llama-server"
    assert bin_path.exists()


def test_llama_server_manager_properties():
    mgr = LlamaServerManager(host="127.0.0.1", port=8099, n_parallel=8)
    assert mgr.base_url == "http://127.0.0.1:8099/v1"
    assert mgr.health_url == "http://127.0.0.1:8099/health"
    assert mgr.is_healthy() is False
    status = mgr.get_status()
    assert status["running"] is False
    assert status["port"] == 8099
    assert status["slots"] == 8
    assert status["binary_installed"] is True


def test_llama_server_is_healthy_mock():
    mgr = LlamaServerManager(port=8099)
    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok"}
        mock_get.return_value = mock_resp

        assert mgr.is_healthy() is True

        mock_resp.json.return_value = {"status": "loading model"}
        assert mgr.is_healthy() is False


def test_download_llama_server_skips_if_existing(tmp_path):
    fake_bin = tmp_path / "llama-server"
    fake_bin.touch(mode=0o755)
    with patch("llama_server_manager.httpx.Client") as mock_client:
        result = download_llama_server(dest_dir=tmp_path)
        assert result == fake_bin
        mock_client.assert_not_called()
