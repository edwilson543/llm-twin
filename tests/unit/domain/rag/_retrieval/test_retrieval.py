from llm_twin.domain.rag import _retrieval
from testing.factories import vectors as vector_factories
from testing.helpers import config as config_helpers
from testing.helpers import storage as storage_helpers


class TestRetrieveRelevantDocuments:
    def test_gets_one_document_most_relevant_to_query(self):
        article = vector_factories.EmbeddedArticleChunk()
        excluded_article = vector_factories.EmbeddedArticleChunk()

        documents = [article, excluded_article]
        db = storage_helpers.InMemoryVectorDatabase(vectors=documents)

        query = "some query"
        config = config_helpers.get_retrieval_config(
            db=db, max_documents_per_query=1, number_of_query_expansions=3
        )

        result = _retrieval.retrieve_relevant_documents(query=query, config=config)

        assert result == [article]

    def test_gets_multiple_documents_relevant_to_query(self):
        article = vector_factories.EmbeddedArticleChunk()
        repository = vector_factories.EmbeddedRepositoryChunk()

        documents = [article, repository]
        db = storage_helpers.InMemoryVectorDatabase(vectors=documents)

        query = "some query"
        config = config_helpers.get_retrieval_config(
            db=db, max_documents_per_query=2, number_of_query_expansions=3
        )

        result = _retrieval.retrieve_relevant_documents(query=query, config=config)

        assert result == documents
