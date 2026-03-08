"""
QueryAnalyzer – classifies an incoming question and decides how it should be routed.

Routing decisions
─────────────────
  history   → The answer can be derived from recent conversation context.
  rag       → The answer requires retrieval from the curated rare-disease corpus.
  web       → The answer requires live web search (recent news, clinical trials, etc.).
  mcp       → The answer requires a structured database query (PubMed, OMIM, Orphanet …).
  hybrid    → A combination of web + RAG is needed.
  requery   → The question is ambiguous; a clarifying rewrite is generated first.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..utils.logger import get_logger

logger = get_logger(__name__)

Route = Literal["history", "rag", "web", "mcp", "hybrid", "requery"]


@dataclass
class AnalysisResult:
    route: Route
    confidence: float
    reasoning: str
    rewritten_query: str | None = None
    sub_queries: List[str] = field(default_factory=list)
    is_medical_emergency: bool = False
    disease_entities: List[str] = field(default_factory=list)


_SYSTEM_PROMPT = """You are a query routing agent for a rare disease Q&A system serving
patients, caregivers, and clinicians.

Your job is to analyse an incoming question and decide which information source is best
suited to answer it.  You MUST respond with valid JSON only – no markdown fences, no
extra keys.

Routes available
────────────────
• "history"  – The question references something already discussed in this conversation
               (e.g. "What did you say about my last question?", follow-up clarifications).
• "rag"      – The question asks about general rare-disease knowledge well covered by the
               curated medical corpus (pathophysiology, diagnosis criteria, treatments,
               patient support resources).
• "web"      – The question requires up-to-date information: ongoing clinical trials,
               recent news, newly approved drugs, support group events, etc.
• "mcp"      – The question needs structured database lookups: specific gene IDs (OMIM),
               Orphanet disease codes, PubMed literature search, ClinicalTrials.gov IDs.
• "hybrid"   – Both live web data AND curated corpus retrieval are needed.
• "requery"  – The question is too vague, contradictory, or domain-irrelevant; generate a
               clarifying rewrite before routing.

Response schema (strict JSON)
──────────────────────────────
{
  "route": "<one of the routes above>",
  "confidence": <0.0–1.0>,
  "reasoning": "<1-2 sentence rationale>",
  "rewritten_query": "<clarified query string or null>",
  "sub_queries": ["<optional decomposed sub-query>"],
  "is_medical_emergency": <true|false>,
  "disease_entities": ["<disease name>"]
}

Safety rule: if the question indicates an acute medical emergency (e.g. severe bleeding,
difficulty breathing, chest pain), set "is_medical_emergency": true regardless of route.
"""


class QueryAnalyzer:
    """LLM-powered query classifier and rewriter."""

    def __init__(self, llm: ChatOpenAI | None = None, config: dict | None = None):
        self.config = config or {}
        self.llm = llm or ChatOpenAI(
            model=self.config.get("llm", {}).get("model", "gpt-4o-mini"),
            temperature=0.0,
        )
        self._confidence_threshold: float = (
            self.config.get("planning", {}).get("route_confidence_threshold", 0.6)
        )

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def analyze(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
    ) -> AnalysisResult:
        """Classify *query* and return an :class:`AnalysisResult`."""
        logger.info("QueryAnalyzer: analysing query → '%s'", query[:120])

        history_context = self._format_history(conversation_history or [])
        user_msg = f"Conversation so far:\n{history_context}\n\nNew question: {query}"

        response = self.llm.invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_msg)]
        )
        result = self._parse_response(response.content, query)

        if result.confidence < self._confidence_threshold and result.route != "requery":
            logger.warning(
                "Low confidence (%.2f) for route '%s'; escalating to requery",
                result.confidence,
                result.route,
            )
            result.route = "requery"
            if result.rewritten_query is None:
                result.rewritten_query = self._generate_clarification(query)

        logger.info(
            "Route decided: %s (confidence=%.2f)", result.route, result.confidence
        )
        return result

    def rewrite_query(self, query: str, feedback: str) -> str:
        """Generate an improved query given failure feedback from a previous attempt."""
        prompt = (
            f"The following rare-disease question could not be answered well:\n\n"
            f"Original question: {query}\n"
            f"Failure reason: {feedback}\n\n"
            f"Rewrite the question to be more specific and answerable. "
            f"Return only the rewritten question, no explanation."
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _parse_response(self, raw: str, original_query: str) -> AnalysisResult:
        try:
            cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
            data = json.loads(cleaned)
            return AnalysisResult(
                route=data.get("route", "rag"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                rewritten_query=data.get("rewritten_query"),
                sub_queries=data.get("sub_queries", []),
                is_medical_emergency=bool(data.get("is_medical_emergency", False)),
                disease_entities=data.get("disease_entities", []),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error("Failed to parse QueryAnalyzer response: %s", exc)
            return AnalysisResult(
                route="rag",
                confidence=0.5,
                reasoning="Fallback: parse error; defaulting to RAG route.",
            )

    def _format_history(self, history: list[dict]) -> str:
        if not history:
            return "(no prior conversation)"
        lines = []
        for turn in history[-6:]:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")[:300]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _generate_clarification(self, query: str) -> str:
        prompt = (
            f"Rewrite this ambiguous rare-disease question to be clearer and more "
            f"specific so a medical knowledge base can answer it:\n\n{query}\n\n"
            f"Return only the rewritten question."
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
