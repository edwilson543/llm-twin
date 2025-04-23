from llm_twin.orchestration.steps.dataset_generation import _fetch_chunked_documents
from testing.factories import documents as document_factories
from testing.factories import vectors as vector_factories
from testing.helpers import config as config_helpers
from testing.helpers import storage as storage_helpers
from testing.helpers import zenml as zenml_helpers


def test_fetches_chunked_documents_from_database():
    author = document_factories.Author()
    other_author = document_factories.Author()

    article_chunk = vector_factories.ArticleChunk(author=author)
    other_article_chunk = vector_factories.ArticleChunk(author=author)
    repository_chunk = vector_factories.RepositoryChunk(author=author)

    # Some chunk that should get filtered out.
    other_author_chunk = vector_factories.ArticleChunk(author=other_author)

    vectors = [article_chunk, other_article_chunk, repository_chunk, other_author_chunk]
    vector_db = storage_helpers.InMemoryVectorDatabase(vectors=vectors)
    context = zenml_helpers.FakeContext()

    with config_helpers.install_in_memory_vector_db(db=vector_db):
        chunks = _fetch_chunked_documents.fetch_chunked_documents.entrypoint(
            author_full_name=author.full_name, batch_size=2, context=context
        )

    assert chunks == [article_chunk, other_article_chunk, repository_chunk]
    assert context.output_metadata["chunked_documents"]["num_chunks"] == 3
    assert context.output_metadata["chunked_documents"]["author"] == author.full_name
