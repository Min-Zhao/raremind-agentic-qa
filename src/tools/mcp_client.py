"""
MCPClient – client for Model Context Protocol (MCP) servers.

Supports both a live MCP HTTP server and mock/local fallbacks for development.
MCP tools exposed:
  • pubmed_search        – PubMed literature search (Entrez API)
  • clinicaltrials_search – ClinicalTrials.gov open studies
  • omim_lookup          – OMIM gene/phenotype records
  • orphanet_lookup      – Orphanet rare disease database
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests

from ..utils.logger import get_logger

logger = get_logger(__name__)

# ── Entrez / NCBI base URLs (used in local fallback) ─────────
_ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_CLINTRIALS_BASE = "https://clinicaltrials.gov/api/v2/studies"


class MCPClient:
    """
    Calls MCP tool endpoints or falls back to direct API calls.

    Parameters
    ----------
    config : dict
        Pipeline configuration containing the ``mcp`` section.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        mcp_cfg = self.config.get("mcp", {})
        self.enabled: bool = mcp_cfg.get("enabled", False)
        self.server_url: str = mcp_cfg.get("server_url", "http://localhost:8765")
        self.available_tools: List[str] = mcp_cfg.get("tools", [])
        self.timeout: int = self.config.get("web", {}).get("timeout_seconds", 10)
        self._session = requests.Session()

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """
        Call a named MCP tool.  Falls back to direct API if MCP server unavailable.

        Parameters
        ----------
        tool_name : str
            One of: pubmed_search, clinicaltrials_search, omim_lookup, orphanet_lookup.
        params : dict
            Tool-specific parameters.

        Returns
        -------
        Tool result (list or dict) or None on failure.
        """
        if self.enabled and tool_name in self.available_tools:
            try:
                return self._mcp_call(tool_name, params)
            except Exception as exc:
                logger.warning(
                    "MCPClient: MCP server call failed (%s); using direct API fallback.", exc
                )

        return self._direct_api_fallback(tool_name, params)

    # ──────────────────────────────────────────────────────────
    # MCP server call
    # ──────────────────────────────────────────────────────────

    def _mcp_call(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """POST to the MCP server's tool endpoint."""
        url = f"{self.server_url}/tools/{tool_name}"
        response = self._session.post(url, json=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    # ──────────────────────────────────────────────────────────
    # Direct API fallbacks
    # ──────────────────────────────────────────────────────────

    def _direct_api_fallback(self, tool_name: str, params: Dict[str, Any]) -> Any:
        dispatch = {
            "pubmed_search": self._pubmed_search,
            "clinicaltrials_search": self._clinicaltrials_search,
            "omim_lookup": self._omim_lookup,
            "orphanet_lookup": self._orphanet_lookup,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            logger.error("MCPClient: unknown tool '%s'.", tool_name)
            return None
        try:
            return fn(params)
        except Exception as exc:
            logger.error("MCPClient: direct API call for '%s' failed: %s", tool_name, exc)
            return None

    def _pubmed_search(self, params: Dict[str, Any]) -> List[Dict[str, str]]:
        query = params.get("query", "")
        max_results = int(params.get("max_results", 5))

        # Step 1: esearch – get PMIDs
        esearch_resp = self._session.get(
            f"{_ENTREZ_BASE}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance",
            },
            timeout=self.timeout,
        )
        esearch_resp.raise_for_status()
        pmids = esearch_resp.json().get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return []

        # Step 2: efetch – get abstracts
        efetch_resp = self._session.get(
            f"{_ENTREZ_BASE}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "json",
                "rettype": "abstract",
            },
            timeout=self.timeout,
        )
        efetch_resp.raise_for_status()

        articles = efetch_resp.json().get("PubmedArticleSet", {}).get("PubmedArticle", [])
        if isinstance(articles, dict):
            articles = [articles]

        results = []
        for art in articles[:max_results]:
            citation = art.get("MedlineCitation", {})
            article = citation.get("Article", {})
            title = article.get("ArticleTitle", "No title")
            abstract_text = article.get("Abstract", {}).get("AbstractText", "")
            if isinstance(abstract_text, list):
                abstract_text = " ".join(str(a) for a in abstract_text)
            pmid = citation.get("PMID", {})
            if isinstance(pmid, dict):
                pmid = pmid.get("#text", "")
            results.append(
                {
                    "source": "PubMed",
                    "pmid": str(pmid),
                    "title": str(title),
                    "abstract": str(abstract_text)[:500],
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
            )
        return results

    def _clinicaltrials_search(self, params: Dict[str, Any]) -> List[Dict[str, str]]:
        condition = params.get("condition", "")
        status = params.get("status", "recruiting")

        resp = self._session.get(
            _CLINTRIALS_BASE,
            params={
                "query.cond": condition,
                "filter.overallStatus": status.upper(),
                "pageSize": 5,
                "format": "json",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        studies = resp.json().get("studies", [])

        results = []
        for study in studies:
            proto = study.get("protocolSection", {})
            id_module = proto.get("identificationModule", {})
            status_module = proto.get("statusModule", {})
            desc_module = proto.get("descriptionModule", {})
            results.append(
                {
                    "source": "ClinicalTrials.gov",
                    "nct_id": id_module.get("nctId", ""),
                    "title": id_module.get("briefTitle", ""),
                    "status": status_module.get("overallStatus", ""),
                    "summary": desc_module.get("briefSummary", "")[:300],
                    "url": f"https://clinicaltrials.gov/study/{id_module.get('nctId', '')}",
                }
            )
        return results

    def _omim_lookup(self, params: Dict[str, Any]) -> Optional[Dict[str, str]]:
        # OMIM requires an API key; return a structured placeholder for now.
        # In production, set OMIM_API_KEY env var and call api.omim.org
        disease_name = params.get("disease_name", "")
        logger.info("MCPClient: OMIM lookup for '%s' (API key required for live data).", disease_name)
        return {
            "source": "OMIM",
            "disease": disease_name,
            "note": (
                "OMIM lookup requires an API key. "
                "Register at https://omim.org/api and set OMIM_API_KEY."
            ),
        }

    def _orphanet_lookup(self, params: Dict[str, Any]) -> Optional[Dict[str, str]]:
        disease_name = params.get("disease_name", "")
        logger.info(
            "MCPClient: Orphanet lookup for '%s' (SPARQL endpoint).", disease_name
        )
        # Orphanet provides a public SPARQL endpoint (data.orphanet.fr).
        # This is a structured placeholder; integrate SPARQL queries in production.
        return {
            "source": "Orphanet",
            "disease": disease_name,
            "note": (
                "Orphanet data available at https://www.orphadata.com/rare-diseases/ "
                "and via SPARQL at https://data.orphanet.fr/sparql"
            ),
        }
