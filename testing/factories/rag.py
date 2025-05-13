import factory

from llm_twin.domain.rag import _query_expansion
from llm_twin.domain import rag

from testing.helpers import models as models_helpers
from testing.helpers import storage as storage_helpers

class RAGConfig(factory.Factory):
    db = factory.LazyFunction(storage_helpers.InMemoryVectorDatabase)

    language_model = factory.LazyFunction(models_helpers.FakeLanguageModel)
    embedding_model = factory.LazyFunction(models_helpers.get_fake_embedding_model)
    # TODO -> SETUP FAKE CE model
    cross_encoder_model = factory.LazyFunction(models_helpers.FakeLanguageModel)

    number_of_query_expansions = 1
    max_chunks_per_query = 1

    class Meta:
        model = rag.RAGConfig


class Expansion(factory.Factory):
    query = factory.Sequence(lambda n: f"query-{n}")
    expansions = factory.List(["query reworded", "query expanded"])

    class Meta:
        model = _query_expansion.Expansion
