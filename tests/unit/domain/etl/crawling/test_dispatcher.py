import pytest

from llm_twin.domain.etl.crawling import _crawlers, _dispatcher


class TestGetCrawler:
    def test_gets_crawler_registered_for_domain(self):
        fake_crawler = _crawlers.FakeCrawler()
        crawler_registry = {"fake.com": fake_crawler}
        dispatcher = _dispatcher.CrawlerDispatcher(crawler_registry=crawler_registry)

        result = dispatcher.get_crawler(link="https://fake.com/edwilson543/posts/")

        assert result is fake_crawler

    def test_raises_when_no_crawler_is_registered_for_domain(self):
        crawler_registry = {"fake.com": _crawlers.FakeCrawler()}
        dispatcher = _dispatcher.CrawlerDispatcher(crawler_registry=crawler_registry)

        with pytest.raises(_dispatcher.NoCrawlerRegisteredForDomain) as exc:
            dispatcher.get_crawler(
                link="https://some-other-site.com/edwilson543/posts/"
            )

        assert exc.value.domain == "some-other-site.com"
