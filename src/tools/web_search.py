"""
WebSearchTool – wraps multiple search providers (SerpAPI, DuckDuckGo, Tavily)
and a lightweight page content extractor.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from ..utils.logger import get_logger

logger = get_logger(__name__)


class WebSearchTool:
    """
    Unified web search and page extraction tool.

    Supports SerpAPI, DuckDuckGo (no key needed), and Tavily.
    Falls back gracefully if a provider is unavailable.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        web_cfg = self.config.get("web", {})
        self.provider: str = web_cfg.get("search_provider", "duckduckgo")
        self.serpapi_key: str = web_cfg.get("serpapi_key", "")
        self.tavily_key: str = web_cfg.get("tavily_key", "")
        self.timeout: int = web_cfg.get("timeout_seconds", 10)
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "RareMindBot/1.0 (rare disease research assistant)"}
        )

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search the web and return a list of result dicts.

        Each dict has keys: title, url, snippet.
        """
        logger.debug("WebSearchTool: searching '%s' via %s", query[:80], self.provider)
        try:
            if self.provider == "serpapi" and self.serpapi_key:
                return self._serpapi_search(query, max_results)
            elif self.provider == "tavily" and self.tavily_key:
                return self._tavily_search(query, max_results)
            else:
                return self._duckduckgo_search(query, max_results)
        except Exception as exc:
            logger.error("WebSearchTool: search failed (%s); returning empty.", exc)
            return []

    def extract_page(self, url: str, max_chars: int = 3000) -> Optional[str]:
        """
        Fetch a web page and extract its main text content.

        Returns None if extraction fails or the domain is blocked.
        """
        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove noise elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator=" ", strip=True)
            # Collapse whitespace
            text = " ".join(text.split())
            logger.debug("WebSearchTool: extracted %d chars from %s", len(text), url)
            return text[:max_chars] if text else None
        except Exception as exc:
            logger.warning("WebSearchTool: page extraction failed for %s (%s)", url, exc)
            return None

    # ──────────────────────────────────────────────────────────
    # Provider implementations
    # ──────────────────────────────────────────────────────────

    def _serpapi_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        params = {
            "q": query,
            "api_key": self.serpapi_key,
            "num": max_results,
            "hl": "en",
        }
        resp = self._session.get(
            "https://serpapi.com/search.json", params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("organic_results", [])[:max_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
            )
        return results

    def _tavily_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        payload = {
            "api_key": self.tavily_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        resp = self._session.post(
            "https://api.tavily.com/search", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("results", [])[:max_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                }
            )
        return results

    def _duckduckgo_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Lightweight DuckDuckGo Instant Answer API (no key, limited results)."""
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(
                        {
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "snippet": r.get("body", ""),
                        }
                    )
            return results
        except ImportError:
            logger.warning(
                "duckduckgo-search not installed. "
                "Install with: pip install duckduckgo-search"
            )
            return self._ddg_html_fallback(query, max_results)

    def _ddg_html_fallback(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Very minimal fallback that returns empty results."""
        logger.warning("WebSearchTool: no search provider available; returning empty.")
        return []
