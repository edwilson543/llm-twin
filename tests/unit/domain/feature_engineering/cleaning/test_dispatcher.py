import pytest

from llm_twin.domain.feature_engineering.cleaning import _cleaners, _dispatcher
from llm_twin.domain.storage import document as document_storage
from testing.factories import documents as document_factories


class TestGetCleaner:
    def test_gets_cleaner_registered_for_article_documents(self):
        article = document_factories.Article()
        dispatcher = _dispatcher.CleanerDispatcher()

        result = dispatcher.get_cleaner(document=article)

        assert isinstance(result, _cleaners.ArticleCleaner)

    def test_gets_cleaner_registered_for_repository_documents(self):
        repository = document_factories.Repository()
        dispatcher = _dispatcher.CleanerDispatcher()

        result = dispatcher.get_cleaner(document=repository)

        assert isinstance(result, _cleaners.RepositoryCleaner)

    def test_raises_when_no_cleaner_is_registered_for_document_type(self):
        author = document_factories.Author()
        dispatcher = _dispatcher.CleanerDispatcher()

        with pytest.raises(_dispatcher.NoCleanerRegistered) as exc:
            dispatcher.get_cleaner(document=author)

        assert exc.value.collection == document_storage.Collection.AUTHORS
