from __future__ import annotations

import dataclasses
from urllib.parse import urlparse

from . import _crawlers


@dataclasses.dataclass(frozen=True)
class NoCrawlerRegisteredForDomain(Exception):
    domain: str


class CrawlerDispatcher:
    def __init__(
        self, *, crawler_registry: dict[str, _crawlers.Crawler] | None = None
    ) -> None:
        self._crawler_registry = crawler_registry or {
            "fake.com": _crawlers.FakeCrawler(),
            "github.com": _crawlers.GithubCrawler(),
        }

    def get_crawler(self, *, link: str) -> _crawlers.Crawler:
        domain = urlparse(link).netloc

        try:
            return self._crawler_registry[domain]
        except KeyError as exc:
            raise NoCrawlerRegisteredForDomain(domain=domain) from exc
