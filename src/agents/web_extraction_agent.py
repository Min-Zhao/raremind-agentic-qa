"""
WebExtractionAgent – retrieves live information from the web and structured biomedical
databases (via MCP) for rare disease queries.

Responsibilities
────────────────
• search_and_extract()  – performs web search and scrapes/summarises page content.
• mcp_query()           – calls MCP-wrapped biomedical APIs (PubMed, OMIM, Orphanet, etc.).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..tools.mcp_client import MCPClient
from ..tools.web_search import WebSearchTool
from ..utils.logger import get_logger

logger = get_logger(__name__)

_SUMMARISE_PROMPT = """You are a medical information specialist helping rare disease patients.

Below are raw web excerpts about: "{query}"

Excerpts:
{excerpts}

Summarise the most relevant, accurate, and patient-friendly information.
Focus on: diagnosis, treatment options, clinical trials, support resources.
Cite which source each piece of information came from.
Be honest about uncertainty. Do NOT fabricate information.
"""


class WebExtractionAgent:
    """
    Searches the web and structured biomedical databases for rare-disease information.

    Parameters
    ----------
    config : dict
        Pipeline configuration.
    llm : optional
        LangChain LLM for content summarisation.
    """

    def __init__(self, config: dict | None = None, llm: ChatOpenAI | None = None):
        self.config = config or {}
        self.llm = llm or ChatOpenAI(
            model=self.config.get("llm", {}).get("model", "gpt-4o-mini"),
            temperature=0.1,
        )
        self.web_tool = WebSearchTool(config=self.config)
        self.mcp_client = MCPClient(config=self.config)
        self.trusted_domains: List[str] = self.config.get("web", {}).get(
            "trusted_domains", []
        )
        self.max_results: int = self.config.get("web", {}).get("max_search_results", 5)

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def search_and_extract(
        self,
        query: str,
        disease_entities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform a web search, scrape page content, and return structured evidence.

        Returns
        -------
        dict with keys: pages, snippets, summary, sources
        """
        # Augment the search query with disease context
        search_query = self._build_search_query(query, disease_entities)
        logger.info("WebExtractionAgent: searching → '%s'", search_query[:120])

        search_results = self.web_tool.search(search_query, max_results=self.max_results)

        if not search_results:
            logger.warning("WebExtractionAgent: no search results found.")
            return self._empty_result()

        # Optionally prioritise trusted domains
        results = self._prioritise_trusted(search_results)

        pages: List[Dict[str, str]] = []
        snippets: List[str] = []
        sources: List[str] = []

        for item in results[: self.max_results]:
            url = item.get("url", "")
            title = item.get("title", "")
            snippet = item.get("snippet", "")

            if snippet:
                snippets.append(f"[{title}] {snippet}")
                sources.append(url)

            # Attempt deeper page extraction
            page_content = self.web_tool.extract_page(url)
            if page_content:
                pages.append({"url": url, "title": title, "content": page_content[:2000]})

        # Summarise all content with LLM
        summary = self._summarise(query, snippets, pages)

        return {
            "pages": pages,
            "snippets": snippets,
            "summary": summary,
            "sources": sources,
            "search_query": search_query,
        }

    def mcp_query(
        self,
        query: str,
        disease_entities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Query structured biomedical databases via MCP tools.

        Returns
        -------
        dict with keys: records, summary, sources
        """
        if not self.config.get("mcp", {}).get("enabled", False):
            logger.info("WebExtractionAgent: MCP disabled; falling back to web search.")
            return self.search_and_extract(query, disease_entities)

        logger.info("WebExtractionAgent: MCP query → '%s'", query[:120])
        entities = disease_entities or []

        records: List[Dict[str, Any]] = []
        sources: List[str] = []

        # PubMed literature search
        pubmed_results = self.mcp_client.call_tool(
            "pubmed_search",
            {"query": f"{query} {' '.join(entities)}", "max_results": 5},
        )
        if pubmed_results:
            records.extend(pubmed_results)
            sources.append("PubMed")

        # ClinicalTrials.gov search
        trials = self.mcp_client.call_tool(
            "clinicaltrials_search",
            {"condition": " ".join(entities) or query, "status": "recruiting"},
        )
        if trials:
            records.extend(trials)
            sources.append("ClinicalTrials.gov")

        # OMIM / Orphanet lookup for named diseases
        for entity in entities[:3]:
            omim = self.mcp_client.call_tool("omim_lookup", {"disease_name": entity})
            if omim:
                records.append(omim)
                sources.append("OMIM")

            orphanet = self.mcp_client.call_tool(
                "orphanet_lookup", {"disease_name": entity}
            )
            if orphanet:
                records.append(orphanet)
                sources.append("Orphanet")

        summary = self._summarise_records(query, records)
        return {"records": records, "summary": summary, "sources": list(set(sources))}

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _build_search_query(
        self, query: str, disease_entities: Optional[List[str]]
    ) -> str:
        entities_str = " ".join(disease_entities or [])
        if entities_str and entities_str.lower() not in query.lower():
            return f"{query} {entities_str} rare disease"
        return f"{query} rare disease"

    def _prioritise_trusted(
        self, results: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        if not self.trusted_domains:
            return results
        trusted, other = [], []
        for r in results:
            domain = urlparse(r.get("url", "")).netloc
            if any(td in domain for td in self.trusted_domains):
                trusted.append(r)
            else:
                other.append(r)
        return trusted + other

    def _summarise(
        self,
        query: str,
        snippets: List[str],
        pages: List[Dict[str, str]],
    ) -> str:
        if not snippets and not pages:
            return "No web content found for this query."

        excerpts = "\n\n".join(snippets)
        for page in pages:
            excerpts += f"\n\n--- {page['title']} ({page['url']}) ---\n{page['content'][:500]}"

        prompt = _SUMMARISE_PROMPT.format(query=query, excerpts=excerpts[:6000])
        try:
            response = self.llm.invoke(
                [SystemMessage(content="You are a medical information specialist."),
                 HumanMessage(content=prompt)]
            )
            return response.content.strip()
        except Exception as exc:
            logger.error("WebExtractionAgent: summarisation failed: %s", exc)
            return "\n".join(snippets[:3])

    def _summarise_records(self, query: str, records: List[Dict[str, Any]]) -> str:
        if not records:
            return "No structured database records found."
        record_text = "\n\n".join(
            str(r) for r in records[:10]
        )
        prompt = (
            f"Summarise these biomedical database records for the question: '{query}'\n\n"
            f"{record_text[:5000]}\n\n"
            f"Provide a concise, patient-friendly summary with key findings."
        )
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as exc:
            logger.error("WebExtractionAgent: record summarisation failed: %s", exc)
            return record_text[:500]

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "pages": [],
            "snippets": [],
            "summary": "No web results found.",
            "sources": [],
            "search_query": "",
        }
