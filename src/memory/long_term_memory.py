"""
LongTermMemory – persistent JSON-backed store for facts and insights accumulated
across multiple sessions.  Complements the short-term ConversationMemory.

Use cases
─────────
• Remembering patient-specific preferences across sessions.
• Caching expensive LLM-generated summaries.
• Storing frequently asked questions and validated answers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class LongTermMemory:
    """
    JSON-file backed long-term key-value memory with optional TTL expiration.

    Parameters
    ----------
    store_path : str or Path
        Path to the JSON file used for persistence.
    ttl_seconds : int, optional
        Time-to-live for each entry in seconds.  Entries older than this
        are considered stale and excluded from retrieval.  None = no expiry.
    """

    def __init__(
        self,
        store_path: str | Path = "./data/long_term_memory.json",
        ttl_seconds: Optional[int] = None,
    ):
        self.store_path = Path(store_path)
        self.ttl_seconds = ttl_seconds
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ──────────────────────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────────────────────

    def set(self, key: str, value: Any, tags: List[str] | None = None) -> None:
        """Store *value* under *key* with optional *tags* and a timestamp."""
        self._data[key] = {
            "value": value,
            "tags": tags or [],
            "timestamp": time.time(),
        }
        self._persist()

    def delete(self, key: str) -> bool:
        """Remove *key* from the store.  Returns True if key existed."""
        if key in self._data:
            del self._data[key]
            self._persist()
            return True
        return False

    def clear(self) -> None:
        """Wipe all stored entries."""
        self._data.clear()
        self._persist()

    # ──────────────────────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """Retrieve the value for *key*, or None if missing/expired."""
        entry = self._data.get(key)
        if entry is None:
            return None
        if self._is_expired(entry):
            logger.debug("LongTermMemory: entry '%s' expired; removing.", key)
            del self._data[key]
            self._persist()
            return None
        return entry["value"]

    def get_by_tags(self, tags: List[str]) -> Dict[str, Any]:
        """Return all non-expired entries whose tag list intersects *tags*."""
        result = {}
        tag_set = set(tags)
        for key, entry in list(self._data.items()):
            if self._is_expired(entry):
                continue
            if tag_set & set(entry.get("tags", [])):
                result[key] = entry["value"]
        return result

    def all_keys(self) -> List[str]:
        """Return all non-expired keys."""
        return [
            k for k, v in self._data.items() if not self._is_expired(v)
        ]

    # ──────────────────────────────────────────────────────────
    # Persistence helpers
    # ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self.store_path.exists():
            try:
                with self.store_path.open("r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.debug(
                    "LongTermMemory: loaded %d entries from '%s'.",
                    len(self._data),
                    self.store_path,
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "LongTermMemory: could not load '%s' (%s). Starting fresh.",
                    self.store_path,
                    exc,
                )
                self._data = {}

    def _persist(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with self.store_path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        if self.ttl_seconds is None:
            return False
        age = time.time() - entry.get("timestamp", 0)
        return age > self.ttl_seconds

    def __len__(self) -> int:
        return len(self.all_keys())

    def __repr__(self) -> str:
        return f"LongTermMemory(entries={len(self)}, path='{self.store_path}')"
