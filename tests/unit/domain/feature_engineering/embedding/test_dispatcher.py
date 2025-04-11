import pytest

from llm_twin.domain.feature_engineering.embedding import _dispatcher, _documents
from llm_twin.domain.storage import vector as vector_storage
from testing.factories import vectors as vector_factories
from testing.helpers import embeddings as embeddings_helpers


class TestEmbedChunk:
    def test_embeds_article_chunk(self):
        article = vector_factories.ArticleChunk()

        embedding_model = embeddings_helpers.get_fake_embedding_model()
        dispatcher = _dispatcher.EmbedderDispatcher(embedding_model=embedding_model)

        embedded_article = dispatcher.embed_chunk(chunk=article)

        assert isinstance(embedded_article, _documents.EmbeddedArticleChunk)
        assert embedded_article.embedding == embedding_model.canned_embedding
        assert embedded_article.content == article.content
        assert embedded_article.raw_document_id == article.raw_document_id

    def test_embeds_repository_chunk(self):
        repository = vector_factories.RepositoryChunk()

        embedding_model = embeddings_helpers.get_fake_embedding_model()
        dispatcher = _dispatcher.EmbedderDispatcher(embedding_model=embedding_model)

        embedded_repository = dispatcher.embed_chunk(chunk=repository)

        assert isinstance(embedded_repository, _documents.EmbeddedRepositoryChunk)
        assert embedded_repository.embedding == embedding_model.canned_embedding
        assert embedded_repository.content == repository.content
        assert embedded_repository.raw_document_id == repository.raw_document_id

    def test_raises_when_no_embedder_is_registered_for_data_category(self):
        some_chunk = vector_factories.Vector()

        embedding_model = embeddings_helpers.get_fake_embedding_model()
        dispatcher = _dispatcher.EmbedderDispatcher(embedding_model=embedding_model)

        with pytest.raises(_dispatcher.NoChunkEmbedderRegistered) as exc:
            dispatcher.embed_chunk(chunk=some_chunk)

        assert exc.value.data_category == vector_storage.DataCategory.TESTING
