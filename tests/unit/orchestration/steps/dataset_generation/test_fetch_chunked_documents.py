from llm_twin.orchestration.steps.dataset_generation import _fetch_chunked_documents
from testing.factories import vectors as vector_factories
from testing.helpers import config as config_helpers
from testing.helpers import context as context_helpers
from testing.helpers import storage as storage_helpers


def test_fetches_chunked_documents_from_database():
    article_chunk = vector_factories.ArticleChunk()
    other_article_chunk = vector_factories.ArticleChunk()
    repository_chunk = vector_factories.RepositoryChunk()

    vector_db = storage_helpers.InMemoryVectorDatabase(
        vectors=[article_chunk, other_article_chunk, repository_chunk]
    )
    context = context_helpers.FakeContext()

    with config_helpers.install_in_memory_vector_db(db=vector_db):
        chunks = _fetch_chunked_documents.fetch_chunked_documents.entrypoint(
            batch_size=2, context=context
        )

    assert chunks == vector_db.vectors
    assert context.output_metadata["fetch_chunked_documents"]["num_chunks"] == 3
