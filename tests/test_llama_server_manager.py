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


def test_download_llama_server_verifies_checksum(tmp_path, monkeypatch):
    import hashlib
    import io
    import tarfile
    import httpx
    import pytest
    import llama_server_manager as lsm

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"#!/bin/sh\n"
        info = tarfile.TarInfo("llama-b1/llama-server")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    archive = buf.getvalue()

    real_client = httpx.Client
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=archive))
    monkeypatch.setattr(lsm.httpx, "Client", lambda **kw: real_client(transport=transport))

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        download_llama_server(dest_dir=tmp_path / "bad", download_url="http://x/a.tar.gz", sha256="0" * 64)
    assert not (tmp_path / "bad" / "llama-server").exists()

    good = download_llama_server(
        dest_dir=tmp_path / "ok", download_url="http://x/a.tar.gz", sha256=hashlib.sha256(archive).hexdigest()
    )
    assert good.read_bytes() == b"#!/bin/sh\n"


def test_prompt_cache_ram_is_capped():
    mgr = LlamaServerManager(cache_ram_mib=512)
    assert mgr.cache_ram_mib == 512
    assert LlamaServerManager().cache_ram_mib <= 2048  # never llama-server's 8 GiB default


def test_backend_selection_and_gpu_flags(monkeypatch):
    import llama_server_manager as lsm

    monkeypatch.delenv("LLAMA_BACKEND", raising=False)
    monkeypatch.setattr(lsm, "gpu_available", lambda: False)
    assert lsm.use_vulkan() is False  # e.g. inside a VM without /dev/dri
    monkeypatch.setattr(lsm, "gpu_available", lambda: True)
    assert lsm.use_vulkan() is True
    monkeypatch.setenv("LLAMA_BACKEND", "cpu")
    assert lsm.use_vulkan() is False

    mgr = LlamaServerManager(n_threads=8, n_threads_batch=12)
    gpu_cmd = mgr.build_command(Path("llama-server"), Path("m.gguf"), vulkan=True)
    cpu_cmd = mgr.build_command(Path("llama-server"), Path("m.gguf"), vulkan=False)
    assert gpu_cmd[gpu_cmd.index("-ngl") + 1] == "99" and "-ngl" not in cpu_cmd
    assert cpu_cmd[cpu_cmd.index("-t") + 1] == "8" and cpu_cmd[cpu_cmd.index("-tb") + 1] == "12"


def test_external_server_url():
    import pytest

    mgr = LlamaServerManager(external_url="http://127.0.0.1:9/v1/")
    assert mgr.base_url == "http://127.0.0.1:9/v1"
    assert mgr.health_url == "http://127.0.0.1:9/health"
    assert mgr.get_status()["backend"] == "external"
    with pytest.raises(RuntimeError, match="not reachable"):
        mgr.ensure_server(Path("unused.gguf"))  # never starts a local process for an external URL
    mgr.stop()  # nothing to stop
    assert mgr.process is None
