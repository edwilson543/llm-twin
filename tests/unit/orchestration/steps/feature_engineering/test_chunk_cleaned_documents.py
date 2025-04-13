from llm_twin.domain.feature_engineering import chunking
from llm_twin.orchestration.steps.feature_engineering import (
    _chunk_cleaned_documents,
)
from testing.factories import vectors as vector_factories
from testing.helpers import config as config_helpers
from testing.helpers import context as context_helpers


def test_chunks_article_documents():
    max_length = 2000
    article = vector_factories.CleanedArticle(content="A? " * max_length * 5)

    context = context_helpers.FakeContext()

    with config_helpers.install_fake_embedding_model():
        chunks = _chunk_cleaned_documents.chunk_cleaned_documents.entrypoint(
            cleaned_documents=[article], context=context
        )

    num_chunks = len("A? ") * 5
    assert len(chunks) == num_chunks
    assert all(isinstance(chunk, chunking.ArticleChunk) for chunk in chunks)
    assert context.output_metadata["chunked_documents"] == {
        "num_documents": 1,
        "num_chunks": num_chunks,
    }


def test_chunks_repository_documents():
    repository = vector_factories.CleanedRepository(content="abc")
    other_repository = vector_factories.CleanedRepository(content="def")
    cleaned_documents = [repository, other_repository]

    context = context_helpers.FakeContext()

    with config_helpers.install_fake_embedding_model():
        chunks = _chunk_cleaned_documents.chunk_cleaned_documents.entrypoint(
            cleaned_documents=cleaned_documents, context=context
        )

    # The fake embedding model just chunks on each character.
    num_chunks = len("abc") + len("def")
    assert len(chunks) == num_chunks
    assert all(isinstance(chunk, chunking.RepositoryChunk) for chunk in chunks)
    assert context.output_metadata["chunked_documents"] == {
        "num_documents": 2,
        "num_chunks": num_chunks,
    }
