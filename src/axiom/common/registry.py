"""Tiny name->object registry so configs can select implementations by string."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from axiom.common.errors import ConfigError

T = TypeVar("T")


class Registry(Generic[T]):
    """Decorator-based registry. Used for answer matchers, reward terms, etc."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if name in self._items:
                raise ConfigError(f"{self._kind} '{name}' already registered")
            self._items[name] = obj
            return obj

        return deco

    def get(self, name: str) -> T:
        if name not in self._items:
            raise ConfigError(f"unknown {self._kind} '{name}'; have {sorted(self._items)}")
        return self._items[name]
