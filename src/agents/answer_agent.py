"""
AnswerAgent – synthesises a final, grounded answer from all gathered evidence.

The agent receives evidence from one or more upstream agents (RAG chunks, web summaries,
MCP records, conversation history) and produces a single coherent, citation-backed
response tailored to rare disease patients and families.

Design principles
─────────────────
• Faithfulness  – every claim must be traceable to provided evidence.
• Clarity       – use plain language; avoid unexplained medical jargon.
• Safety        – include appropriate caveats; never replace professional advice.
• Transparency  – cite sources inline; flag uncertainty explicitly.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are RareMind, an expert AI assistant specialized in rare diseases,
particularly rare lymphatic and vascular disorders such as Complex Lymphatic Anomalies (CLA),
Gorham-Stout Disease, Generalized Lymphatic Anomaly, Kaposiform Lymphangiomatosis, and related
conditions.

Your audience is patients, caregivers, and families dealing with rare diseases. They need:
• Accurate, evidence-based information
• Clear, compassionate, jargon-free explanations
• Honest acknowledgement of uncertainty or knowledge gaps
• Guidance toward appropriate medical professionals and support resources

CRITICAL SAFETY RULES:
1. NEVER replace or contradict a patient's doctor's advice.
2. ALWAYS recommend consulting a specialist for treatment decisions.
3. If you are uncertain, say so explicitly.
4. For emergencies, always direct to emergency services first.

When citing sources, use the format [Source: <name>] inline."""

_ANSWER_PROMPT = """Use the evidence below to answer the patient's question.

Question: {query}
{rewritten_note}

Evidence
────────
{evidence_text}

Conversation context (recent turns):
{history_text}

Instructions:
- Synthesise a clear, compassionate, evidence-based answer.
- Cite sources inline (e.g., [Source: PubMed], [Source: Orphanet]).
- If evidence is limited or conflicting, acknowledge this openly.
- End with a brief "📋 Key Takeaways" bullet list (max 3 bullets).
- Suggest relevant next steps or resources if appropriate.
- Do NOT fabricate information not present in the evidence.

Answer:"""


class AnswerAgent:
    """
    Final answer synthesiser.

    Parameters
    ----------
    llm : optional
        A pre-built LangChain chat model.
    config : dict, optional
        Pipeline configuration.
    """

    def __init__(self, llm: ChatOpenAI | None = None, config: dict | None = None):
        self.config = config or {}
        self.llm = llm or ChatOpenAI(
            model=self.config.get("llm", {}).get("model", "gpt-4o"),
            temperature=0.1,
            max_tokens=self.config.get("llm", {}).get("max_tokens", 2048),
        )

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def synthesize(
        self,
        query: str,
        evidence: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        rewritten_query: Optional[str] = None,
        disease_entities: Optional[List[str]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Produce a final answer and a list of source citations.

        Parameters
        ----------
        query : str
            The original user query.
        evidence : dict
            Aggregated evidence from RAG, web, MCP, and history agents.
        conversation_history : list[dict], optional
            Recent conversation turns for context.
        rewritten_query : str, optional
            A clarified version of the query (if requery was needed).
        disease_entities : list[str], optional
            Named disease entities for targeted response framing.

        Returns
        -------
        (answer_text, sources_list)
        """
        logger.info("AnswerAgent: synthesising answer for query='%s'", query[:120])

        evidence_text, sources = self._format_evidence(evidence)
        history_text = self._format_history(conversation_history or [])

        rewritten_note = (
            f"\n(Note: The question was clarified to: '{rewritten_query}')"
            if rewritten_query and rewritten_query != query
            else ""
        )

        prompt = _ANSWER_PROMPT.format(
            query=query,
            rewritten_note=rewritten_note,
            evidence_text=evidence_text,
            history_text=history_text,
        )

        if not evidence_text.strip() or evidence_text.strip() == "No evidence available.":
            answer = self._no_evidence_response(query)
        else:
            response = self.llm.invoke(
                [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            answer = response.content.strip()

        logger.info(
            "AnswerAgent: generated answer (%d chars, %d sources).",
            len(answer),
            len(sources),
        )
        return answer, sources

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _format_evidence(
        self, evidence: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Convert the raw evidence dict into a single text block + source list."""
        sections: List[str] = []
        sources: List[Dict[str, Any]] = []

        # ── History evidence ─────────────────────────────────
        history_ev = evidence.get("history", {})
        if history_ev.get("found") and history_ev.get("answer"):
            sections.append(
                f"[Conversation History]\n{history_ev['answer']}"
            )
            sources.append({"type": "history", "label": "Previous conversation"})

        # ── RAG evidence ─────────────────────────────────────
        rag_ev = evidence.get("rag", {})
        chunks = rag_ev.get("chunks", [])
        if chunks:
            rag_lines = [f"[Rare Disease Corpus – {c.get('source', 'doc')}]" for c in chunks]
            chunk_texts = "\n\n".join(
                f"Source: {c.get('source', 'unknown')} (score={c.get('score', 0):.2f})\n"
                f"{c.get('content', '')}"
                for c in chunks
            )
            sections.append(f"[Knowledge Base Retrieval]\n{chunk_texts}")
            for c in chunks:
                sources.append({
                    "type": "rag",
                    "label": c.get("source", "Medical corpus"),
                    "score": c.get("score", 0),
                    "disease": c.get("disease"),
                })

        # ── Web evidence ─────────────────────────────────────
        web_ev = evidence.get("web", {})
        if web_ev.get("summary"):
            sections.append(f"[Web Search Summary]\n{web_ev['summary']}")
            for src in web_ev.get("sources", []):
                sources.append({"type": "web", "label": src})

        # ── MCP evidence ─────────────────────────────────────
        mcp_ev = evidence.get("mcp", {})
        if mcp_ev.get("summary"):
            sections.append(f"[Biomedical Database (MCP)]\n{mcp_ev['summary']}")
            for src in mcp_ev.get("sources", []):
                sources.append({"type": "mcp", "label": src})

        if not sections:
            return "No evidence available.", []

        return "\n\n" + "\n\n".join(sections), sources

    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        if not history:
            return "(No prior conversation)"
        lines = []
        for turn in history[-4:]:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")[:400]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _no_evidence_response(query: str) -> str:
        return (
            f"I wasn't able to find specific information about this question in my "
            f"knowledge base or current web sources.\n\n"
            f"**What I recommend:**\n"
            f"- Consult a specialist in rare lymphatic/vascular diseases.\n"
            f"- Visit [NORD (National Organization for Rare Disorders)](https://rarediseases.org) "
            f"or [NIH Genetic and Rare Diseases Information Center](https://rarediseases.info.nih.gov).\n"
            f"- Contact disease-specific patient advocacy organizations.\n\n"
            f"*This AI assistant has limited knowledge of very specific or newly emerging "
            f"topics. A rare disease specialist can provide the most accurate guidance.*"
        )
