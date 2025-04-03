import factory

from llm_twin.domain.storage import vector as vector_storage


class _Vector(vector_storage.Vector):
    name: str

    class _Config(vector_storage.Config):
        collection = vector_storage.Collection.TESTING_VECTORS
        category = vector_storage.DataCategory.TESTING


class Vector(factory.Factory):
    name = factory.Sequence(lambda n: f"vector-{n}")

    class Meta:
        model = _Vector


class _VectorEmbedding(vector_storage.VectorEmbedding):
    name: str

    class _Config(vector_storage.Config):
        collection = vector_storage.Collection.TESTING_VECTOR_EMBEDDINGS
        category = vector_storage.DataCategory.TESTING


class VectorEmbedding(factory.Factory):
    name = factory.Sequence(lambda n: f"vector-{n}")
    embedding = factory.LazyFunction(lambda: [1.0, 0.0, 0.0])

    class Meta:
        model = _VectorEmbedding
