import factory

from llm_twin.domain.rag import _query_expansion


class Expansion(factory.Factory):
    query = factory.Sequence(lambda n: f"query-{n}")
    expansions = factory.List(["query reworded", "query expanded"])

    class Meta:
        model = _query_expansion.Expansion
