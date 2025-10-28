from drl_system.config import WebToolsConfig
from drl_system.integration.web_tools import WebScraper, WebSessionManager


HTML_SNIPPET = """
<html>
  <head><title>Example Page</title></head>
  <body>
    <p>Hello <strong>world</strong>.</p>
    <a href="https://example.com/about">About</a>
    <a href="https://other.com/contact">Contact</a>
  </body>
</html>
"""


def test_web_scraper_extracts_links_and_summary():
    scraper = WebScraper()
    result = scraper.scrape(HTML_SNIPPET)
    assert result["title"] == "Example Page"
    assert "https://example.com/about" in result["links"]
    summary = result["summary"]
    assert "Hello" in summary


def test_web_session_manager_domain_filters_and_history():
    config = WebToolsConfig(allowed_domains=["example.com"], max_concurrent_requests=2)

    class DummyManager(WebSessionManager):
        def fetch(self, url: str, timeout: float = 5.0) -> str:  # type: ignore[override]
            assert self.is_domain_allowed(url)
            with self._lock:
                self._history.append(url)
            return f"payload:{url}"

    manager = DummyManager(config)
    urls = ["https://example.com/a", "https://example.com/b"]
    payloads = manager.fetch_many(urls)
    assert sorted(payloads) == sorted([f"payload:{url}" for url in urls])
    assert manager.history() == urls
    assert not manager.is_domain_allowed("https://forbidden.org")
