"""OpenAI-compatible teacher client.

One implementation, two uses: optional teacher-trace generation (this module) and the
commonsense judge (prm/labeling/judge.py imports `TeacherClient`). Credentials come from
the env var named in the config; nothing is hardcoded.
"""

from __future__ import annotations

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from omegaconf import DictConfig

from axiom.common.errors import ConfigError
from axiom.common.logging import get_logger

log = get_logger(__name__)


class TeacherClient:
    """Minimal chat-completions client over a configurable endpoint."""

    def __init__(self, cfg: DictConfig) -> None:
        self._key = os.environ.get(cfg.api_key_env, "")
        if not self._key:
            raise ConfigError(f"env var {cfg.api_key_env} is empty; set it in .env")
        self._url = f"{cfg.api_base.rstrip('/')}/v1/chat/completions"
        self._model = cfg.model
        self._max_tokens = cfg.max_tokens
        self._temperature = cfg.temperature
        self._timeout = cfg.request_timeout_s
        self._workers = cfg.max_concurrency

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Single chat completion; returns the assistant message text."""
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body = json.dumps(
            {
                "model": self._model,
                "messages": messages,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
            }
        ).encode()
        req = urllib.request.Request(
            self._url,
            data=body,
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            payload = json.loads(resp.read())
        return payload["choices"][0]["message"]["content"]

    def complete_batch(self, prompts: list[str], system: str | None = None) -> list[str]:
        """Concurrent completions, order preserved. Failures become empty strings (logged)."""

        def _safe(p: str) -> str:
            try:
                return self.complete(p, system)
            except Exception as exc:
                log.warning("teacher call failed: %s", exc)
                return ""

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            return list(pool.map(_safe, prompts))
