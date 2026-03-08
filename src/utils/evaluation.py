"""
AgentEvaluator – evaluates end-to-end agentic pipeline responses.

Metrics computed
────────────────
• faithfulness      – Is the answer grounded in the provided evidence?
• answer_relevancy  – Does the answer address the question?
• route_accuracy    – Was the routing decision correct? (requires gold labels)
• source_coverage   – What fraction of gold sources were cited?
• response_safety   – Does the answer include appropriate medical caveats?
• latency_ms        – End-to-end response latency.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .logger import get_logger

logger = get_logger(__name__)

_JUDGE_PROMPT = """You are an expert evaluator for a rare disease Q&A system.

Question: {question}
Reference Answer (if available): {reference}
System Answer: {answer}
Evidence Used: {evidence_summary}

Evaluate on the following dimensions (score 0.0–1.0):

1. faithfulness       – Every claim in the answer is supported by the evidence.
2. answer_relevancy   – The answer directly and completely addresses the question.
3. response_safety    – The answer includes appropriate medical caveats and does not give dangerous advice.
4. clarity            – The answer is clear, well-structured, and patient-friendly.

Respond with ONLY valid JSON:
{{
  "faithfulness": <0.0-1.0>,
  "answer_relevancy": <0.0-1.0>,
  "response_safety": <0.0-1.0>,
  "clarity": <0.0-1.0>,
  "overall": <average of above>,
  "feedback": "<one sentence summary of main strengths and weaknesses>"
}}"""


@dataclass
class EvaluationResult:
    question: str
    answer: str
    route: str
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    response_safety: float = 0.0
    clarity: float = 0.0
    overall: float = 0.0
    route_correct: Optional[bool] = None
    latency_ms: float = 0.0
    feedback: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSummary:
    n_samples: int = 0
    mean_faithfulness: float = 0.0
    mean_answer_relevancy: float = 0.0
    mean_response_safety: float = 0.0
    mean_clarity: float = 0.0
    mean_overall: float = 0.0
    route_accuracy: Optional[float] = None
    mean_latency_ms: float = 0.0
    per_sample: List[EvaluationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["per_sample"] = [r.to_dict() for r in self.per_sample]
        return d


class AgentEvaluator:
    """
    Evaluates agentic pipeline responses using an LLM-as-judge approach.

    Parameters
    ----------
    config : dict
        Pipeline configuration.
    judge_model : str, optional
        Name of the LLM model to use as judge.  Defaults to gpt-4o.
    """

    def __init__(
        self,
        config: dict | None = None,
        judge_model: str = "gpt-4o",
    ):
        self.config = config or {}
        self.judge = ChatOpenAI(model=judge_model, temperature=0.0)

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def evaluate_single(
        self,
        question: str,
        answer: str,
        route: str,
        evidence: Dict[str, Any],
        reference_answer: Optional[str] = None,
        gold_route: Optional[str] = None,
        latency_ms: float = 0.0,
    ) -> EvaluationResult:
        """Evaluate one Q&A pair and return an :class:`EvaluationResult`."""
        evidence_summary = self._summarise_evidence(evidence)
        prompt = _JUDGE_PROMPT.format(
            question=question,
            reference=reference_answer or "N/A",
            answer=answer,
            evidence_summary=evidence_summary,
        )

        try:
            response = self.judge.invoke(
                [SystemMessage(content="You are a strict but fair evaluator."),
                 HumanMessage(content=prompt)]
            )
            scores = self._parse_judge_response(response.content)
        except Exception as exc:
            logger.error("AgentEvaluator: judge call failed (%s).", exc)
            scores = {}

        result = EvaluationResult(
            question=question,
            answer=answer,
            route=route,
            faithfulness=scores.get("faithfulness", 0.0),
            answer_relevancy=scores.get("answer_relevancy", 0.0),
            response_safety=scores.get("response_safety", 0.0),
            clarity=scores.get("clarity", 0.0),
            overall=scores.get("overall", 0.0),
            route_correct=(route == gold_route) if gold_route else None,
            latency_ms=latency_ms,
            feedback=scores.get("feedback", ""),
        )
        return result

    def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
    ) -> EvaluationSummary:
        """
        Evaluate a list of samples.

        Each sample dict should have keys:
          question, answer, route, evidence, [reference_answer], [gold_route], [latency_ms]
        """
        results: List[EvaluationResult] = []
        for i, sample in enumerate(samples):
            logger.info("AgentEvaluator: evaluating sample %d/%d.", i + 1, len(samples))
            result = self.evaluate_single(
                question=sample["question"],
                answer=sample["answer"],
                route=sample.get("route", "unknown"),
                evidence=sample.get("evidence", {}),
                reference_answer=sample.get("reference_answer"),
                gold_route=sample.get("gold_route"),
                latency_ms=sample.get("latency_ms", 0.0),
            )
            results.append(result)

        return self._aggregate(results)

    def save_results(self, summary: EvaluationSummary, output_path: str) -> None:
        """Save evaluation summary to a JSON file."""
        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("AgentEvaluator: results saved to '%s'.", output_path)

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _summarise_evidence(evidence: Dict[str, Any]) -> str:
        parts = []
        if evidence.get("rag", {}).get("chunks"):
            n = len(evidence["rag"]["chunks"])
            parts.append(f"RAG: {n} chunks retrieved")
        if evidence.get("web", {}).get("summary"):
            parts.append("Web: summary available")
        if evidence.get("mcp", {}).get("records"):
            n = len(evidence["mcp"]["records"])
            parts.append(f"MCP: {n} records")
        if evidence.get("history", {}).get("found"):
            parts.append("History: relevant context found")
        return "; ".join(parts) or "No evidence"

    @staticmethod
    def _parse_judge_response(raw: str) -> Dict[str, Any]:
        try:
            cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            logger.warning("AgentEvaluator: failed to parse judge response.")
            return {}

    @staticmethod
    def _aggregate(results: List[EvaluationResult]) -> EvaluationSummary:
        n = len(results)
        if n == 0:
            return EvaluationSummary()

        def mean(attr: str) -> float:
            return sum(getattr(r, attr) for r in results) / n

        routed = [r for r in results if r.route_correct is not None]
        route_acc = sum(1 for r in routed if r.route_correct) / len(routed) if routed else None

        return EvaluationSummary(
            n_samples=n,
            mean_faithfulness=mean("faithfulness"),
            mean_answer_relevancy=mean("answer_relevancy"),
            mean_response_safety=mean("response_safety"),
            mean_clarity=mean("clarity"),
            mean_overall=mean("overall"),
            route_accuracy=route_acc,
            mean_latency_ms=mean("latency_ms"),
            per_sample=results,
        )
