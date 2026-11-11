"""Install Ollama for the current user without root (CPU runtime only).

Streams the official Linux archive, keeps the binary and CPU libraries, and skips
the multi-gigabyte GPU runtimes that a CPU-only machine never loads.

Usage: uv run python -m scripts.install_ollama_local [--prefix .local/ollama]
Then:  .local/ollama/bin/ollama serve
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import httpx
import zstandard

URL = "https://ollama.com/download/ollama-linux-amd64.tar.zst"
SKIP = ("cuda", "rocm", "vulkan", "jetpack", "mlx")


class _Stream:
    """File-like adapter over an HTTP byte stream, for streaming decompression."""

    def __init__(self, response: httpx.Response) -> None:
        self._chunks = response.iter_bytes(1 << 20)
        self._buffer = b""

    def read(self, size: int = -1) -> bytes:
        while size < 0 or len(self._buffer) < size:
            chunk = next(self._chunks, None)
            if chunk is None:
                break
            self._buffer += chunk
        out, self._buffer = (
            (self._buffer, b"") if size < 0 else (self._buffer[:size], self._buffer[size:])
        )
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=Path(".local/ollama"))
    args = parser.parse_args()
    args.prefix.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    with httpx.stream("GET", URL, follow_redirects=True, timeout=None) as response:
        response.raise_for_status()
        reader = zstandard.ZstdDecompressor().stream_reader(_Stream(response))
        with tarfile.open(fileobj=reader, mode="r|") as archive:
            for member in archive:
                if any(token in member.name.lower() for token in SKIP):
                    skipped += 1
                    continue
                archive.extract(member, args.prefix, filter="data")
                kept += 1
    binary = args.prefix / "bin" / "ollama"
    print(f"extracted {kept} entries (skipped {skipped} GPU-runtime entries) -> {binary}")


if __name__ == "__main__":
    main()
