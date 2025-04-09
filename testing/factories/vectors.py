import factory

from llm_twin.domain.feature_engineering import chunking, cleaning
from llm_twin.domain.storage import vector as vector_storage

from . import documents


# Dummy.


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


# Cleaning.


class _CleanedDocument(factory.Factory):
    content = factory.Sequence(lambda n: f"content-{n}")
    platform = "some-platform"

    author = factory.SubFactory(documents.Author)
    author_id = factory.LazyAttribute(lambda o: o.author.id)
    author_full_name = factory.LazyAttribute(lambda o: o.author.full_name)


class CleanedArticle(_CleanedDocument):
    link = factory.Sequence(lambda n: f"https://fake.com/article-{n}/")

    class Meta:
        model = cleaning.CleanedArticle


class CleanedRepository(_CleanedDocument):
    name = factory.Sequence(lambda n: f"repo-{n}")
    link = factory.Sequence(lambda n: f"https://github.com/edwilson543/repo-{n}/")

    class Meta:
        model = cleaning.CleanedRepository


# Chunking.


class _Chunk(_CleanedDocument):
    cleaned_document_id = factory.Sequence(lambda n: f"cleaned-document-{n}")


class ArticleChunk(_Chunk):
    link = factory.Sequence(lambda n: f"https://fake.com/article-{n}/")

    class Meta:
        model = chunking.ArticleChunk


class RepositoryChunk(_Chunk):
    name = factory.Sequence(lambda n: f"repo-{n}")
    link = factory.Sequence(lambda n: f"https://github.com/edwilson543/repo-{n}/")

    class Meta:
        model = chunking.RepositoryChunk
