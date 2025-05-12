from llm_twin.domain.rag import _query_expansion
from testing.helpers import models as models_helpers


class TestExpandQuery:
    def test_expands_query_into_multiple_queries(self):
        language_model = models_helpers.FakeLanguageModel()

        expansion = _query_expansion.expand_query(
            query="some query",
            number_of_query_expansions=2,
            language_model=language_model,
        )

        assert len(expansion.expansions) == 2
        assert len(expansion.all_queries) == 3
