"""Web browsing and scraping helpers with conservative defaults."""
from __future__ import annotations

import html
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from html.parser import HTMLParser
from threading import Lock
from typing import Dict, Iterable, List, Optional
from urllib import request
from urllib.parse import urlparse

from ..config import WebToolsConfig


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[str] = []
        self.title: Optional[str] = None
        self._capture_title = False

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str]]) -> None:
        if tag.lower() == "a":
            for key, value in attrs:
                if key.lower() == "href":
                    self.links.append(value)
        if tag.lower() == "title":
            self._capture_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._capture_title = False

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self.title = (self.title or "") + data.strip()


@dataclass
class WebScraper:
    """Utility for extracting structured information from HTML."""

    user_agent: str = "SelfImprovingDRL/1.0"

    def scrape(self, html_text: str) -> Dict[str, object]:
        parser = _LinkParser()
        parser.feed(html_text)
        summary = self.summarize(html_text)
        return {
            "title": parser.title or "",
            "links": parser.links,
            "summary": summary,
        }

    @staticmethod
    def summarize(html_text: str, max_chars: int = 280) -> str:
        text = HTMLStripper.strip_tags(html_text)
        text = " ".join(text.split())
        if len(text) > max_chars:
            return text[: max_chars - 1] + "…"
        return text

    @staticmethod
    def extract_links(html_text: str) -> List[str]:
        parser = _LinkParser()
        parser.feed(html_text)
        return parser.links


class HTMLStripper(HTMLParser):
    """Simple HTML to text converter used for summaries."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._parts.append(html.unescape(data))

    def get_data(self) -> str:
        return " ".join(self._parts)

    @classmethod
    def strip_tags(cls, html_text: str) -> str:
        parser = cls()
        parser.feed(html_text)
        return parser.get_data()


@dataclass
class WebSessionManager:
    """Executes web requests with domain restrictions and threading support."""

    config: WebToolsConfig
    _history: List[str] = field(default_factory=list, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)

    def is_domain_allowed(self, url: str) -> bool:
        if not self.config.allowed_domains:
            return True
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        return any(hostname.endswith(domain) for domain in self.config.allowed_domains)

    def fetch(self, url: str, timeout: float = 5.0) -> str:
        if not (self.config.enable_browsing or self.config.enable_scraping):
            raise RuntimeError("Web access disabled by configuration")
        if not self.is_domain_allowed(url):
            raise PermissionError(f"Domain not allowed: {url}")
        req = request.Request(url, headers={"User-Agent": self.config.user_agent})
        with request.urlopen(req, timeout=timeout) as response:
            charset = response.headers.get_content_charset("utf-8")
            payload = response.read().decode(charset, errors="replace")
        with self._lock:
            self._history.append(url)
        return payload

    def fetch_many(self, urls: Iterable[str], timeout: float = 5.0) -> List[str]:
        if not urls:
            return []
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests) as executor:
            futures = [executor.submit(self.fetch, url, timeout=timeout) for url in urls]
            done, pending = wait(futures)
            for future in pending:
                future.cancel()
            return [future.result() for future in done]

    def history(self) -> List[str]:
        with self._lock:
            return list(self._history)


__all__ = ["WebScraper", "WebSessionManager", "HTMLStripper"]
