"""
PlanningAgent – the central orchestrator of the agentic rare disease QA pipeline.

Execution flow
──────────────
  1. Receive user query + conversation history.
  2. Call QueryAnalyzer → decide route (history / rag / web / mcp / hybrid / requery).
  3. If route == "requery": rewrite query and loop (max N attempts).
  4. Dispatch to the appropriate specialist agent(s).
  5. Pass gathered evidence to AnswerAgent → synthesise final answer.
  6. Return a structured AgentResponse with reasoning trace.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..memory.conversation_memory import ConversationMemory
from ..utils.logger import get_logger
from .answer_agent import AnswerAgent
from .history_agent import HistoryAgent
from .query_analyzer import AnalysisResult, QueryAnalyzer
from .rag_agent import RAGAgent
from .web_extraction_agent import WebExtractionAgent

logger = get_logger(__name__)


@dataclass
class AgentStep:
    """Records one step in the planning trace."""
    step: int
    agent: str
    action: str
    result_summary: str
    duration_ms: float


@dataclass
class AgentResponse:
    query: str
    final_answer: str
    route: str
    confidence: float
    sources: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[AgentStep] = field(default_factory=list)
    is_medical_emergency: bool = False
    disease_entities: List[str] = field(default_factory=list)
    requery_count: int = 0
    total_duration_ms: float = 0.0


class PlanningAgent:
    """
    Master orchestrator that coordinates all specialist agents.

    Parameters
    ----------
    config : dict
        Parsed config.yaml as a dictionary.
    llm : optional
        A pre-built LangChain chat model.  If None, one is constructed from *config*.
    """

    def __init__(self, config: dict, llm=None):
        self.config = config
        self.max_requery = config.get("planning", {}).get("max_requery_attempts", 2)

        self.query_analyzer = QueryAnalyzer(llm=llm, config=config)
        self.history_agent = HistoryAgent(config=config)
        self.rag_agent = RAGAgent(config=config, llm=llm)
        self.web_agent = WebExtractionAgent(config=config, llm=llm)
        self.answer_agent = AnswerAgent(llm=llm, config=config)
        self.memory = ConversationMemory(
            max_turns=config.get("memory", {}).get("max_history_turns", 10)
        )

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def run(self, query: str) -> AgentResponse:
        """
        Execute the full agentic pipeline for *query*.

        Returns an :class:`AgentResponse` containing the final answer and
        the complete reasoning trace.
        """
        t0 = time.perf_counter()
        trace: List[AgentStep] = []
        step_idx = 0
        requery_count = 0
        current_query = query

        logger.info("PlanningAgent: starting pipeline for query='%s'", query[:120])

        # ── Emergency safety check (fast path before LLM calls) ─────────────
        if self._is_emergency_keywords(query):
            return self._emergency_response(query, trace, t0)

        # ── Planning loop (with re-query support) ────────────────────────────
        evidence: Dict[str, Any] = {}
        analysis: Optional[AnalysisResult] = None

        while requery_count <= self.max_requery:
            step_idx += 1
            t_step = time.perf_counter()

            analysis = self.query_analyzer.analyze(
                current_query, self.memory.get_history()
            )
            trace.append(
                AgentStep(
                    step=step_idx,
                    agent="QueryAnalyzer",
                    action=f"Analyzed query → route={analysis.route}",
                    result_summary=(
                        f"Route: {analysis.route} | "
                        f"Confidence: {analysis.confidence:.2f} | "
                        f"{analysis.reasoning}"
                    ),
                    duration_ms=(time.perf_counter() - t_step) * 1000,
                )
            )

            if analysis.is_medical_emergency:
                return self._emergency_response(query, trace, t0)

            if analysis.route == "requery":
                if requery_count >= self.max_requery:
                    logger.warning("Max requery attempts reached; falling back to RAG.")
                    analysis.route = "rag"
                else:
                    requery_count += 1
                    current_query = (
                        analysis.rewritten_query or
                        self.query_analyzer.rewrite_query(
                            current_query, "Query was too ambiguous."
                        )
                    )
                    step_idx += 1
                    trace.append(
                        AgentStep(
                            step=step_idx,
                            agent="PlanningAgent",
                            action="Rewrote ambiguous query",
                            result_summary=f"New query: {current_query}",
                            duration_ms=0,
                        )
                    )
                    logger.info("Requerying (attempt %d): '%s'", requery_count, current_query)
                    continue

            # ── Dispatch to specialist agents ────────────────────────────────
            evidence = self._dispatch(
                route=analysis.route,
                query=current_query,
                analysis=analysis,
                trace=trace,
                step_start=step_idx,
            )
            step_idx = len(trace)
            break  # successful dispatch; exit planning loop

        # ── Answer synthesis ─────────────────────────────────────────────────
        step_idx += 1
        t_step = time.perf_counter()

        final_answer, sources = self.answer_agent.synthesize(
            query=query,
            rewritten_query=current_query if current_query != query else None,
            evidence=evidence,
            conversation_history=self.memory.get_history(),
            disease_entities=analysis.disease_entities if analysis else [],
        )

        trace.append(
            AgentStep(
                step=step_idx,
                agent="AnswerAgent",
                action="Synthesised final answer",
                result_summary=f"Answer length: {len(final_answer)} chars | Sources: {len(sources)}",
                duration_ms=(time.perf_counter() - t_step) * 1000,
            )
        )

        # ── Persist to memory ────────────────────────────────────────────────
        self.memory.add_turn(role="user", content=query)
        self.memory.add_turn(role="assistant", content=final_answer)

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info("PlanningAgent: finished in %.0f ms", total_ms)

        return AgentResponse(
            query=query,
            final_answer=final_answer,
            route=analysis.route if analysis else "rag",
            confidence=analysis.confidence if analysis else 0.5,
            sources=sources,
            trace=trace,
            is_medical_emergency=False,
            disease_entities=analysis.disease_entities if analysis else [],
            requery_count=requery_count,
            total_duration_ms=total_ms,
        )

    def reset_memory(self) -> None:
        """Clear conversation history (start a new session)."""
        self.memory.clear()
        logger.info("Conversation memory cleared.")

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _dispatch(
        self,
        route: str,
        query: str,
        analysis: AnalysisResult,
        trace: List[AgentStep],
        step_start: int,
    ) -> Dict[str, Any]:
        """Route the query to specialist agent(s) and collect evidence."""
        evidence: Dict[str, Any] = {}
        step_idx = step_start

        if route == "history":
            step_idx += 1
            t = time.perf_counter()
            history_result = self.history_agent.retrieve(
                query, self.memory.get_history()
            )
            evidence["history"] = history_result
            trace.append(
                AgentStep(
                    step=step_idx,
                    agent="HistoryAgent",
                    action="Retrieved relevant conversation history",
                    result_summary=history_result.get("summary", "No history found"),
                    duration_ms=(time.perf_counter() - t) * 1000,
                )
            )

        if route in ("rag", "hybrid"):
            step_idx += 1
            t = time.perf_counter()
            rag_result = self.rag_agent.retrieve(
                query,
                sub_queries=analysis.sub_queries,
                disease_entities=analysis.disease_entities,
            )
            evidence["rag"] = rag_result
            trace.append(
                AgentStep(
                    step=step_idx,
                    agent="RAGAgent",
                    action="Retrieved documents from rare-disease corpus",
                    result_summary=(
                        f"Chunks retrieved: {len(rag_result.get('chunks', []))} | "
                        f"Top score: {rag_result.get('top_score', 0):.3f}"
                    ),
                    duration_ms=(time.perf_counter() - t) * 1000,
                )
            )

        if route in ("web", "hybrid"):
            step_idx += 1
            t = time.perf_counter()
            web_result = self.web_agent.search_and_extract(
                query, disease_entities=analysis.disease_entities
            )
            evidence["web"] = web_result
            trace.append(
                AgentStep(
                    step=step_idx,
                    agent="WebExtractionAgent",
                    action="Searched web and extracted relevant content",
                    result_summary=(
                        f"Pages found: {len(web_result.get('pages', []))} | "
                        f"Snippets: {len(web_result.get('snippets', []))}"
                    ),
                    duration_ms=(time.perf_counter() - t) * 1000,
                )
            )

        if route == "mcp":
            step_idx += 1
            t = time.perf_counter()
            mcp_result = self.web_agent.mcp_query(
                query, disease_entities=analysis.disease_entities
            )
            evidence["mcp"] = mcp_result
            trace.append(
                AgentStep(
                    step=step_idx,
                    agent="MCPAgent",
                    action="Queried structured biomedical databases via MCP",
                    result_summary=f"Records retrieved: {len(mcp_result.get('records', []))}",
                    duration_ms=(time.perf_counter() - t) * 1000,
                )
            )

        return evidence

    @staticmethod
    def _is_emergency_keywords(query: str) -> bool:
        emergency_terms = [
            "can't breathe", "cannot breathe", "severe bleeding", "chest pain",
            "heart attack", "stroke", "unconscious", "passing out", "call 911",
            "emergency room", "suicide", "overdose",
        ]
        q_lower = query.lower()
        return any(term in q_lower for term in emergency_terms)

    @staticmethod
    def _emergency_response(
        query: str, trace: List[AgentStep], t0: float
    ) -> AgentResponse:
        answer = (
            "⚠️ **This appears to be a medical emergency.**\n\n"
            "Please **call 911** (US) or your local emergency number immediately, "
            "or go to the nearest emergency room.\n\n"
            "This AI assistant cannot provide emergency medical guidance. "
            "For life-threatening situations, always contact emergency services first."
        )
        logger.warning("Medical emergency detected – returning safety response.")
        return AgentResponse(
            query=query,
            final_answer=answer,
            route="emergency",
            confidence=1.0,
            is_medical_emergency=True,
            trace=trace,
            total_duration_ms=(time.perf_counter() - t0) * 1000,
        )
