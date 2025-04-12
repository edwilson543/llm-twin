import pytest

from llm_twin.domain.feature_engineering.cleaning import _dispatcher, _documents
from llm_twin.domain.storage import document as document_storage
from testing.factories import documents as document_factories


class TestCleanDocument:
    def test_cleans_article_document(self):
        article = document_factories.Article(
            content={"some": "  con", "more": "tent.  "}
        )
        dispatcher = _dispatcher.CleanerDispatcher()

        cleaned_article = dispatcher.clean_document(document=article)

        assert isinstance(cleaned_article, _documents.CleanedArticle)
        assert cleaned_article.content == "con tent."

    def test_cleans_repository_document(self):
        repository = document_factories.Repository(
            content={"line 1:": "def   ", "line 2": "do_something"}
        )
        dispatcher = _dispatcher.CleanerDispatcher()

        cleaned_repository = dispatcher.clean_document(document=repository)

        assert isinstance(cleaned_repository, _documents.CleanedRepository)
        assert cleaned_repository.content == "def do_something"

    def test_raises_when_no_cleaner_is_registered_for_document_type(self):
        author = document_factories.Author()
        dispatcher = _dispatcher.CleanerDispatcher()

        with pytest.raises(_dispatcher.NoCleanerRegistered) as exc:
            dispatcher.clean_document(document=author)

        assert exc.value.collection == document_storage.Collection.AUTHORS
