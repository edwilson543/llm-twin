import pytest

from llm_twin.domain.feature_engineering.chunking import _dispatcher, _documents
from llm_twin.domain.storage import vector as vector_storage
from testing.factories import vectors as vector_factories
from testing.helpers import embeddings as embeddings_helpers


def _get_dispatcher() -> _dispatcher.ChunkerDispatcher:
    embedding_model = embeddings_helpers.get_fake_embedding_model()
    return _dispatcher.ChunkerDispatcher(embedding_model=embedding_model)


class TestSplitDocumentIntoChunks:
    def test_splits_cleaned_article_into_chunks(self):
        content = "A? "
        reps = 5
        max_length = 2000
        article = vector_factories.CleanedArticle(content=content * reps * max_length)

        dispatcher = _get_dispatcher()

        article_chunks = dispatcher.split_document_into_chunks(document=article)

        assert len(article_chunks) == len(content) * reps
        for article_chunk in article_chunks:
            assert isinstance(article_chunk, _documents.ArticleChunk)

    def test_splits_cleaned_repository_into_chunks(self):
        repository = vector_factories.CleanedRepository(content="abc")
        dispatcher = _get_dispatcher()

        repository_chunks = dispatcher.split_document_into_chunks(document=repository)

        # The fake embedding model just chunks on each character.
        assert all(
            isinstance(chunk, _documents.RepositoryChunk) for chunk in repository_chunks
        )
        assert len(repository_chunks) == len("abc")
        assert [chunk.content for chunk in repository_chunks] == ["a", "b", "c"]

    def test_raises_when_no_chunker_is_registered_for_data_category(self):
        some_document = vector_factories.Vector()
        dispatcher = _get_dispatcher()

        with pytest.raises(_dispatcher.NoDocumentChunkerRegistered) as exc:
            dispatcher.split_document_into_chunks(document=some_document)

        assert exc.value.data_category == vector_storage.DataCategory.TESTING
