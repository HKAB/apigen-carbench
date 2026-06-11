"""Loads and indexes the standardized API definitions."""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import APIDef


class ApiLibrary:
    """In-memory registry of APIDefs, indexed by name."""

    def __init__(self, apis: list[APIDef]):
        self._apis = {api.name: api for api in apis}

    @classmethod
    def from_json(cls, path: str | Path) -> "ApiLibrary":
        raw = json.loads(Path(path).read_text())
        return cls([APIDef.model_validate(item) for item in raw])

    def get(self, name: str) -> APIDef | None:
        return self._apis.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._apis

    def __len__(self) -> int:
        return len(self._apis)

    def all(self) -> list[APIDef]:
        return list(self._apis.values())

    def names(self) -> list[str]:
        return list(self._apis)
