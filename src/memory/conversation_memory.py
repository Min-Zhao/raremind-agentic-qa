"""
ConversationMemory – short-term in-session memory that stores conversation turns
and exposes them for history-based retrieval and context injection.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List


class ConversationMemory:
    """
    Sliding-window conversation history store.

    Stores the last *max_turns* user/assistant exchanges in memory.
    Each turn is a dict with keys: ``role`` ("user" | "assistant") and ``content``.

    Parameters
    ----------
    max_turns : int
        Maximum number of turns to retain.  Older turns are discarded.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._history: Deque[Dict[str, str]] = deque(maxlen=max_turns)

    # ──────────────────────────────────────────────────────────
    # Mutation
    # ──────────────────────────────────────────────────────────

    def add_turn(self, role: str, content: str) -> None:
        """Append a new turn.  Automatically evicts the oldest if at capacity."""
        self._history.append({"role": role, "content": content})

    def clear(self) -> None:
        """Wipe all stored history."""
        self._history.clear()

    # ──────────────────────────────────────────────────────────
    # Access
    # ──────────────────────────────────────────────────────────

    def get_history(self) -> List[Dict[str, str]]:
        """Return a chronological list of all stored turns."""
        return list(self._history)

    def get_last_n(self, n: int) -> List[Dict[str, str]]:
        """Return the most recent *n* turns."""
        return list(self._history)[-n:]

    def get_formatted(self, role_labels: Dict[str, str] | None = None) -> str:
        """
        Return a human-readable string of the conversation history.

        Parameters
        ----------
        role_labels : dict, optional
            Maps role names to display labels, e.g. {"user": "Patient", "assistant": "RareMind"}.
        """
        labels = role_labels or {"user": "User", "assistant": "Assistant"}
        lines = []
        for turn in self._history:
            label = labels.get(turn["role"], turn["role"].capitalize())
            lines.append(f"{label}: {turn['content']}")
        return "\n".join(lines)

    def to_langchain_messages(self) -> List:
        """Convert history to LangChain message objects for direct LLM injection."""
        from langchain_core.messages import AIMessage, HumanMessage

        messages = []
        for turn in self._history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))
        return messages

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return f"ConversationMemory(turns={len(self._history)}, max={self.max_turns})"
