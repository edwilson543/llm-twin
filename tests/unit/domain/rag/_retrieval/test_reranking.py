from llm_twin.domain.rag._retrieval import _reranking
from testing.factories import vectors as vector_factories
from testing.helpers import models as models_helpers


class TestRerankDocuments:
    def test_returns_top_ranked_documents_only(self):
        cross_encoder = models_helpers.FakeCrossEncoder()
        documents = [vector_factories.EmbeddedArticleChunk() for _ in range(0, 3)]

        result = _reranking.rerank_documents(
            query="some query",
            documents=documents,
            top_k=2,
            cross_encoder_model=cross_encoder,
        )

        assert result == documents[:2]

    def test_returns_all_documents_when_number_of_documents_less_than_top_k(self):
        cross_encoder = models_helpers.FakeCrossEncoder()
        documents = [vector_factories.EmbeddedArticleChunk() for _ in range(0, 2)]

        result = _reranking.rerank_documents(
            query="some query",
            documents=documents,
            top_k=3,
            cross_encoder_model=cross_encoder,
        )

        assert result == documents
