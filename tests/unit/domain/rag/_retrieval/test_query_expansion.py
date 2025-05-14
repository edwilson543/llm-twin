from llm_twin.domain.rag._retrieval import _query_expansion
from testing.helpers import models as models_helpers


class TestExpansion__FromQuery:
    def test_expands_query_into_multiple_queries(self):
        language_model = models_helpers.FakeLanguageModel()

        expansion = _query_expansion.Expansion.from_query(
            query="some query", n_expansions=2, language_model=language_model
        )

        assert len(expansion.expansions) == 2
        assert len(expansion.all_queries) == 3
