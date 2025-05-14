from llm_twin.domain.rag import _retrieval
from testing.factories import rag as rag_factories
from testing.helpers import storage as storage_helpers
from testing.helpers import storage as storage_helpers
from testing.helpers import config as config_helpers


class TestRetrieveContextForQuery:
    def test_gets_context_from_similar_article_and_repository_chunks(self):
        article = vector_factories.EmbeddedArticleChunk()
        excluded_article = vector_factories.EmbeddedArticleChunk()
        repository = vector_factories.EmbeddedRepositoryChunk()

        vectors = [article, excluded_article, repository]
        db = storage_helpers.InMemoryVectorDatabase(vectors=vectors)

        query = "some query"
        config = config_helpers.get_rag_config(
            db=db, max_chunks_per_query=1, number_of_query_expansions=1
        )

        result = _retrieval.retrieve_context_for_query(query=query, config=config)

        assert result == [article, repository]
