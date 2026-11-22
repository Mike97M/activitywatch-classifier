"""Automatic downloader and process manager for native llama-server (C++ binary).

Provides high-performance, slot-parallel, prefix-cached local LLM inference
with zero Python GIL overhead and 100% self-contained binaries.
"""

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any
import httpx

logger = logging.getLogger(__name__)

DEFAULT_RELEASE_TAG = "b12172"
DEFAULT_DOWNLOAD_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{DEFAULT_RELEASE_TAG}/"
    f"llama-{DEFAULT_RELEASE_TAG}-bin-ubuntu-x64.tar.gz"
)
# From the GitHub release asset digest; update together with DEFAULT_RELEASE_TAG.
DEFAULT_DOWNLOAD_SHA256 = "4496c40c11710b520a745bd9845a47fc91c8d42077c4069130a075d08d4718f1"
BIN_DIR = Path(__file__).resolve().parent / "bin" / "llama-cpp"
SERVER_BIN_NAME = "llama-server"
DEFAULT_PORT = 8088


def get_llama_server_binary() -> Optional[Path]:
    """Find local llama-server binary if available."""
    # 1. Local bin/llama-cpp directory
    local_bin = BIN_DIR / SERVER_BIN_NAME
    if local_bin.is_file() and os.access(local_bin, os.X_OK):
        return local_bin

    # 2. Check direct bin/llama-server
    direct_bin = BIN_DIR.parent / SERVER_BIN_NAME
    if direct_bin.is_file() and os.access(direct_bin, os.X_OK):
        return direct_bin

    # 3. Check system PATH
    sys_bin = shutil.which(SERVER_BIN_NAME)
    if sys_bin:
        return Path(sys_bin)

    return None


def download_llama_server(
    dest_dir: Path = BIN_DIR,
    download_url: Optional[str] = None,
    sha256: Optional[str] = None,
) -> Path:
    """Download, verify (SHA256) and extract the precompiled llama-server C++ binary package.

    The default release is always verified; a custom `download_url` is verified when `sha256` is given.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_bin = dest_dir / SERVER_BIN_NAME
    if target_bin.is_file() and os.access(target_bin, os.X_OK):
        logger.info(f"llama-server already present at {target_bin}")
        return target_bin

    url = download_url or DEFAULT_DOWNLOAD_URL
    expected_sha256 = sha256 or (DEFAULT_DOWNLOAD_SHA256 if url == DEFAULT_DOWNLOAD_URL else None)
    if not expected_sha256:
        logger.warning(f"No SHA256 given for {url}; the downloaded binary will NOT be verified.")
    logger.info(f"Downloading precompiled llama-server from {url}...")

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        with httpx.Client(follow_redirects=True, timeout=120.0) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                downloaded = 0
                digest = hashlib.sha256()
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        f.write(chunk)
                        digest.update(chunk)
                        downloaded += len(chunk)
                        if total > 0 and downloaded % (1024 * 1024 * 5) < 65536:
                            pct = (downloaded / total) * 100
                            logger.info(f"Downloading llama-server: {pct:.1f}% ({downloaded // (1024*1024)}MB / {total // (1024*1024)}MB)")

        if expected_sha256 and digest.hexdigest() != expected_sha256.lower():
            raise RuntimeError(
                f"Checksum mismatch for {url}: expected {expected_sha256}, got {digest.hexdigest()}. "
                "Refusing to install."
            )

        logger.info(f"Extracting llama-server archive into {dest_dir}...")
        with tarfile.open(tmp_path, "r:gz") as tar:
            # Look for member paths
            members = tar.getmembers()
            # If archive has root folder like 'llama-b12172/', strip prefix
            for m in members:
                # Strip leading dir if present
                parts = Path(m.name).parts
                if len(parts) > 1 and parts[0].startswith("llama-"):
                    rel_name = Path(*parts[1:])
                else:
                    rel_name = Path(m.name)

                target_dest = dest_dir / rel_name
                if rel_name.is_absolute() or ".." in rel_name.parts:
                    logger.warning(f"Skipping unsafe archive member: {m.name}")
                    continue
                if m.isdir():
                    target_dest.mkdir(parents=True, exist_ok=True)
                elif m.isfile():
                    target_dest.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(m) as src, open(target_dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    target_dest.chmod(m.mode | 0o755 if "bin" in str(target_dest) or m.name.endswith(".so") or SERVER_BIN_NAME in m.name else m.mode)
                elif m.islnk() or m.issym():
                    # Handle symlink
                    target_dest.parent.mkdir(parents=True, exist_ok=True)
                    if target_dest.is_symlink() or target_dest.exists():
                        target_dest.unlink()
                    try:
                        os.symlink(m.linkname, target_dest)
                    except Exception as e:
                        logger.debug(f"Symlink failed for {m.name}: {e}")

        # Ensure executable
        if target_bin.exists():
            target_bin.chmod(0o755)

        logger.info(f"Successfully installed llama-server at: {target_bin}")
        return target_bin
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


class LlamaServerManager:
    """Manages the lifecycle of a dedicated native llama-server process."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        n_ctx: int = 16384,
        n_parallel: int = 4,
        n_threads: int = 4,
    ):
        self.host = host
        self.port = port
        self.n_ctx = n_ctx
        self.n_parallel = n_parallel
        self.n_threads = n_threads
        self.process: Optional[subprocess.Popen] = None
        self._current_model_path: Optional[Path] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    def is_healthy(self) -> bool:
        """Check if server is responding and ready to accept requests."""
        try:
            with httpx.Client(timeout=2.0) as client:
                r = client.get(self.health_url)
                if r.status_code == 200:
                    data = r.json()
                    status = data.get("status", "")
                    # "ok" or "no slot available" means model is loaded and ready
                    return status in ["ok", "no slot available"]
                return False
        except Exception:
            return False

    def ensure_server(
        self,
        model_path: Path,
        wait_ready: bool = True,
        timeout: int = 45,
    ) -> str:
        """Ensure llama-server binary is present and server is actively running."""
        if self.is_healthy():
            return self.base_url

        binary_path = get_llama_server_binary()
        if not binary_path:
            binary_path = download_llama_server()

        model_path = Path(model_path).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # If a process was previously started by us and died, clean up
        if self.process is not None and self.process.poll() is not None:
            self.process = None

        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "llama-server.log"

        cmd = [
            str(binary_path),
            "-m", str(model_path),
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.n_ctx),
            "-np", str(self.n_parallel),
            "-t", str(self.n_threads),
            "--reasoning", "off",
            "--reasoning-budget", "0",
            "--reasoning-format", "none",
            "--slots",
        ]

        logger.info(f"Starting native llama-server on port {self.port} with {self.n_parallel} slots...")
        logger.info(f"Command: {' '.join(cmd)}")

        with open(log_file, "a") as log_fp:
            self.process = subprocess.Popen(
                cmd,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                cwd=str(binary_path.parent),
                start_new_session=True,
            )

        self._current_model_path = model_path

        if wait_ready:
            start_t = time.time()
            logger.info("Waiting for llama-server to load model and report healthy...")
            while time.time() - start_t < timeout:
                if self.is_healthy():
                    elapsed = time.time() - start_t
                    logger.info(f"llama-server is online and healthy after {elapsed:.1f}s.")
                    return self.base_url
                if self.process.poll() is not None:
                    code = self.process.returncode
                    raise RuntimeError(f"llama-server exited prematurely with code {code}. Check {log_file}")
                time.sleep(0.5)

            raise TimeoutError(f"llama-server did not become healthy within {timeout}s. Check {log_file}")

        return self.base_url

    def stop(self):
        """Stop managed llama-server process."""
        if self.process is not None and self.process.poll() is None:
            logger.info(f"Stopping llama-server (PID {self.process.pid})...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def get_status(self) -> Dict[str, Any]:
        """Return server status metadata."""
        healthy = self.is_healthy()
        binary = get_llama_server_binary()
        return {
            "binary_installed": binary is not None,
            "binary_path": str(binary) if binary else None,
            "running": healthy,
            "port": self.port,
            "base_url": self.base_url,
            "model_path": str(self._current_model_path) if self._current_model_path else None,
            "slots": self.n_parallel,
            "pid": self.process.pid if self.process and self.process.poll() is None else None,
        }


# Global singleton instance
llama_server = LlamaServerManager()
