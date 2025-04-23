from urllib.parse import urlparse

from . import _crawlers


class CrawlerDispatcher:
    def __init__(self) -> None:
        self._crawler_registry = {
            "broken.com": _crawlers.BrokenCrawler(),
            "fake.com": _crawlers.FakeCrawler(),
            "github.com": _crawlers.GithubCrawler(),
        }
        self._fallback = _crawlers.CustomArticleCrawler()

    def get_crawler(self, *, link: str) -> _crawlers.Crawler:
        domain = urlparse(link).netloc

        return self._crawler_registry.get(domain, self._fallback)
