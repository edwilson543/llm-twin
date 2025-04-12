from llm_twin.domain.feature_engineering import embedding
from llm_twin.orchestration.steps.feature_engineering import (
    _embed_chunked_documents,
)
from testing.factories import vectors as vector_factories
from testing.helpers import context as context_helpers
from testing.helpers import settings as settings_helpers


def test_embeds_article_and_repository_documents():
    article_chunk = vector_factories.ArticleChunk()
    other_article_chunk = vector_factories.ArticleChunk(
        raw_document_id=article_chunk.raw_document_id
    )
    repository_chunk = vector_factories.RepositoryChunk()
    chunked_documents = [article_chunk, other_article_chunk, repository_chunk]

    context = context_helpers.FakeContext()

    with (
        settings_helpers.install_fake_embedding_model() as embedding_model,
        settings_helpers.install_in_memory_vector_db() as vector_db,
    ):
        embedded_chunks = _embed_chunked_documents.embed_chunked_documents.entrypoint(
            batch_size=2, chunked_documents=chunked_documents, context=context
        )

    assert vector_db.vectors == embedded_chunks
    assert len(embedded_chunks) == len(chunked_documents)

    embedded_articles = embedded_chunks[:2]
    for embedded_article in embedded_articles:
        assert isinstance(embedded_article, embedding.EmbeddedArticleChunk)
        assert embedded_article.raw_document_id == article_chunk.raw_document_id
        assert embedded_article.embedding == embedding_model.canned_embedding

    embedded_repository = embedded_chunks[2]
    assert isinstance(embedded_repository, embedding.EmbeddedRepositoryChunk)
    assert embedded_repository.raw_document_id == repository_chunk.raw_document_id
    assert embedded_repository.embedding == embedding_model.canned_embedding

    assert context.output_metadata["embedded_chunks"]["num_documents"] == 3
    assert context.output_metadata["embedded_chunks"]["num_batches"] == 2
    assert context.output_metadata["embedded_chunks"]["batch_size"] == 2
