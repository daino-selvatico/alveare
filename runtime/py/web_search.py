"""
Web Search Engine for Alveare.
Provides robust, bot-safe, zero-API-key search functionality across multiple fallback providers:
1. DuckDuckGo HTML parser (with realistic desktop browser headers)
2. Wikipedia OpenSearch & Extracts API (for instant encyclopedic and factual knowledge)
3. DuckDuckGo Instant Answer API (for instant definitions and zero-click answers)
"""

import urllib.request
import urllib.parse
import json
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("alveare.web_search")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
)


class WebSearchEngine:
    """Multi-provider resilient web search engine."""

    def __init__(self, timeout_sec: float = 6.0):
        self.timeout = timeout_sec

    def search(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Executes a search for `query` using resilient fallbacks.
        Returns a dictionary with status, query, results list, and formatted text summary.
        """
        clean_query = query.strip()
        if not clean_query:
            return {
                "status": "error",
                "error": "Query di ricerca vuota.",
                "query": query,
                "results": [],
                "summary": "Nessuna informazione trovata: la query era vuota."
            }

        # 1. Try DuckDuckGo HTML parser first
        results = self._search_duckduckgo_html(clean_query, max_results)

        # 2. If no results or error, try Wikipedia Search
        if not results:
            results = self._search_wikipedia(clean_query, max_results)

        # 3. If still no results, try DuckDuckGo Instant Answer API
        if not results:
            results = self._search_duckduckgo_instant(clean_query, max_results)

        if not results:
            return {
                "status": "ok",
                "query": clean_query,
                "results": [],
                "summary": f"Nessun risultato rilevante trovato per '{clean_query}'."
            }

        summary = self._format_summary(clean_query, results)
        return {
            "status": "ok",
            "query": clean_query,
            "results": results,
            "summary": summary
        }

    def _search_duckduckgo_html(self, query: str, max_results: int) -> List[Dict[str, str]]:
        url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
            "DNT": "1",
            "Connection": "keep-alive"
        }
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            results = []
            # Match each search result item in DDG HTML layout
            # Standard DDG html has class="result__snippet" and class="result__title" or class="result__url"
            snippets = re.findall(r'<a[^>]+class=[\"\']result__snippet[^>]*>(.*?)</a>', html, re.DOTALL)
            title_matches = re.findall(r'<a[^>]+class=[\"\']result__url[^>]*>(.*?)</a>', html, re.DOTALL)
            href_matches = re.findall(r'<a[^>]+class=[\"\']result__url[^>]*href=[\"\'](.*?)[\"\']', html, re.DOTALL)

            for i in range(min(len(snippets), max_results)):
                snippet_raw = snippets[i]
                clean_snippet = re.sub(r'<[^>]+>', '', snippet_raw).strip()
                clean_snippet = re.sub(r'\s+', ' ', clean_snippet)

                title_raw = title_matches[i] if i < len(title_matches) else query
                clean_title = re.sub(r'<[^>]+>', '', title_raw).strip()

                href = href_matches[i] if i < len(href_matches) else ""
                # Unquote DuckDuckGo redirect URL if present
                if "/l/?" in href and "uddg=" in href:
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                    href = parsed.get("uddg", [href])[0]

                if clean_snippet:
                    results.append({
                        "title": clean_title or f"Risultato {i+1}",
                        "snippet": clean_snippet,
                        "url": href
                    })

            return results
        except Exception as e:
            logger.warning(f"DuckDuckGo HTML search fallito per '{query}': {e}")
            return []

    def _search_wikipedia(self, query: str, max_results: int) -> List[Dict[str, str]]:
        url = "https://it.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max_results
        })
        req = urllib.request.Request(url, headers={"User-Agent": "AlveareAssistant/3.0 (SmartHome AI)"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            results = []
            for item in data.get("query", {}).get("search", []):
                snippet = re.sub(r'<[^>]+>', '', item.get("snippet", "")).strip()
                snippet = re.sub(r'\s+', ' ', snippet)
                title = item.get("title", "")
                page_url = f"https://it.wikipedia.org/wiki/{urllib.parse.quote(title)}"
                if snippet:
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": page_url
                    })
            return results
        except Exception as e:
            logger.warning(f"Wikipedia search fallito per '{query}': {e}")
            return []

    def _search_duckduckgo_instant(self, query: str, max_results: int) -> List[Dict[str, str]]:
        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        })
        req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            results = []
            abstract = data.get("AbstractText", "")
            heading = data.get("Heading", "")
            if abstract:
                results.append({
                    "title": heading or query,
                    "snippet": abstract,
                    "url": data.get("AbstractURL", "")
                })

            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "")[:60],
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", "")
                    })
            return results[:max_results]
        except Exception as e:
            logger.warning(f"DuckDuckGo Instant search fallito per '{query}': {e}")
            return []

    def _format_summary(self, query: str, results: List[Dict[str, str]]) -> str:
        lines = [f"Risultati della ricerca web per '{query}':"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            url = r.get("url", "")
            entry = f"[{i}] {title}: {snippet}"
            if url:
                entry += f" (Fonte: {url})"
            lines.append(entry)
        return "\n".join(lines)


# Singleton
_SEARCH_ENGINE: Optional[WebSearchEngine] = None

def get_web_search_engine() -> WebSearchEngine:
    global _SEARCH_ENGINE
    if _SEARCH_ENGINE is None:
        _SEARCH_ENGINE = WebSearchEngine()
    return _SEARCH_ENGINE
