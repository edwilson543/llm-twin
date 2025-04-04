import pytest

from llm_twin.domain.feature_engineering.cleaning import _cleaners, _dispatcher
from llm_twin.domain.storage import document as document_storage
from testing.factories import documents as document_factories


class TestGetCrawler:
    def test_gets_crawler_registered_for_domain(self):
        article = document_factories.Article()
        dispatcher = _dispatcher.CleanerDispatcher()

        result = dispatcher.get_cleaner(document=article)

        assert isinstance(result, _cleaners.ArticleCleaner)

    def test_raises_when_no_crawler_is_registered_for_domain(self):
        author = document_factories.Author()
        dispatcher = _dispatcher.CleanerDispatcher()

        with pytest.raises(_dispatcher.NoCleanerRegistered) as exc:
            dispatcher.get_cleaner(document=author)

        assert exc.value.collection == document_storage.Collection.AUTHORS
