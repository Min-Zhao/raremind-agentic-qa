"""
HistoryAgent – checks whether the current query can be answered from conversation history.

Strategy
────────
1. Embed the query and each past assistant turn.
2. Compute cosine similarity.
3. If similarity > threshold, extract the relevant answer span.
4. Return structured result with confidence score.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ..utils.logger import get_logger

logger = get_logger(__name__)

_EXTRACT_PROMPT = """You are an assistant helping a rare disease patient.
The following is our recent conversation:

{history}

The patient just asked: "{query}"

If the conversation already contains enough information to answer this question, provide
a concise and accurate answer grounded in what was already discussed.
If not, respond with exactly: INSUFFICIENT_HISTORY

Answer:"""


class HistoryAgent:
    """
    Retrieves relevant answers from past conversation turns.

    Parameters
    ----------
    config : dict
        Pipeline configuration dictionary.
    """

    def __init__(self, config: dict | None = None, llm: ChatOpenAI | None = None):
        self.config = config or {}
        self.threshold: float = (
            self.config.get("memory", {}).get("history_relevance_threshold", 0.75)
        )
        self.llm = llm or ChatOpenAI(
            model=self.config.get("llm", {}).get("model", "gpt-4o-mini"),
            temperature=0.0,
        )
        self.embedder = OpenAIEmbeddings(
            model=self.config.get("embedding", {}).get("model", "text-embedding-3-small")
        )

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def retrieve(
        self, query: str, history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Check history for a usable answer.

        Returns
        -------
        dict with keys:
          found (bool), answer (str|None), confidence (float), summary (str)
        """
        if not history:
            return self._not_found("No conversation history available.")

        logger.info("HistoryAgent: checking %d turns for relevant content.", len(history))

        # Embed query
        q_vec = self._embed(query)

        # Collect assistant turns and compute similarity
        scored_turns: List[tuple[float, Dict[str, str]]] = []
        for turn in history:
            if turn.get("role") != "assistant":
                continue
            vec = self._embed(turn["content"])
            sim = self._cosine_similarity(q_vec, vec)
            scored_turns.append((sim, turn))

        if not scored_turns:
            return self._not_found("No assistant turns in history.")

        best_score, best_turn = max(scored_turns, key=lambda x: x[0])
        logger.debug("HistoryAgent: best similarity=%.3f", best_score)

        if best_score < self.threshold:
            return self._not_found(
                f"Best history similarity {best_score:.3f} below threshold {self.threshold}."
            )

        # Ask LLM to extract/confirm the answer from history
        history_text = self._format_history(history)
        prompt = _EXTRACT_PROMPT.format(history=history_text, query=query)
        response = self.llm.invoke(
            [SystemMessage(content="You are a helpful rare disease Q&A assistant."),
             HumanMessage(content=prompt)]
        )
        extracted = response.content.strip()

        if "INSUFFICIENT_HISTORY" in extracted:
            return self._not_found("LLM judged history insufficient.")

        logger.info("HistoryAgent: found relevant answer in history (score=%.3f).", best_score)
        return {
            "found": True,
            "answer": extracted,
            "confidence": best_score,
            "summary": f"Answered from conversation history (similarity={best_score:.2f})",
        }

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _embed(self, text: str) -> List[float]:
        return self.embedder.embed_query(text)

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x ** 2 for x in a))
        mag_b = math.sqrt(sum(x ** 2 for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        lines = []
        for turn in history:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _not_found(reason: str) -> Dict[str, Any]:
        logger.debug("HistoryAgent: %s", reason)
        return {"found": False, "answer": None, "confidence": 0.0, "summary": reason}
