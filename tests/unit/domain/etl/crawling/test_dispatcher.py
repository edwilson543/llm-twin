from llm_twin.domain.etl.crawling import _crawlers, _dispatcher


class TestGetCrawler:
    def test_gets_crawler_registered_for_domain(self):
        dispatcher = _dispatcher.CrawlerDispatcher()

        crawler = dispatcher.get_crawler(link="https://fake.com/edwilson543/posts/")

        assert isinstance(crawler, _crawlers.FakeCrawler)

    def test_raises_when_no_crawler_is_registered_for_domain(self):
        dispatcher = _dispatcher.CrawlerDispatcher()

        crawler = dispatcher.get_crawler(
            link="https://some-other-site.com/edwilson543/posts/"
        )

        assert isinstance(crawler, _crawlers.CustomArticleCrawler)
