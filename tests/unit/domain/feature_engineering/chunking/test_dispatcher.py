import pytest

from llm_twin.domain.feature_engineering.chunking import _chunkers, _dispatcher
from llm_twin.domain.storage import vector as vector_storage
from testing.factories import vectors as vector_factories


class TestGetChunker:
    def test_gets_correct_chunker_for_article_documents(self):
        article = vector_factories.CleanedArticle()
        dispatcher = _dispatcher.ChunkerDispatcher()

        result = dispatcher.get_chunker(document=article)

        assert isinstance(result, _chunkers.ArticleChunker)

    def test_gets_correct_chunker_for_repository_documents(self):
        repository = vector_factories.CleanedRepository()
        dispatcher = _dispatcher.ChunkerDispatcher()

        result = dispatcher.get_chunker(document=repository)

        assert isinstance(result, _chunkers.RepositoryChunker)

    def test_raises_when_no_chunker_is_registered_for_data_category(self):
        some_document = vector_factories.Vector()
        dispatcher = _dispatcher.ChunkerDispatcher()

        with pytest.raises(_dispatcher.NoDocumentChunkerRegistered) as exc:
            dispatcher.get_chunker(document=some_document)

        assert exc.value.data_category == vector_storage.DataCategory.TESTING
