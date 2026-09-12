"""Keep Gradio's requests to its own server off outbound HTTP proxies."""

from __future__ import annotations

import os


def configure_local_proxy_bypass(server_name: str = "127.0.0.1") -> None:
    """Merge both NO_PROXY spellings without changing the outbound proxy settings."""
    entries = []
    seen = set()
    for value in (
        os.environ.get("NO_PROXY", ""),
        os.environ.get("no_proxy", ""),
        "localhost,127.0.0.1,::1",
        server_name.strip("[]"),
    ):
        for entry in value.split(","):
            entry = entry.strip()
            if entry and entry.casefold() not in seen:
                entries.append(entry)
                seen.add(entry.casefold())
    bypass = ",".join(entries)
    os.environ["NO_PROXY"] = bypass
    os.environ["no_proxy"] = bypass
