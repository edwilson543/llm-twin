import factory

from llm_twin.domain.feature_engineering import chunking, cleaning, embedding
from llm_twin.domain.storage import vector as vector_storage

from . import _base, _mixins, documents


# Dummy.


class _Vector(vector_storage.Vector):
    name: str

    class _Config(vector_storage.Config):
        collection = vector_storage.Collection.TESTING_VECTORS
        category = vector_storage.DataCategory.TESTING


class Vector(_base.Factory):
    name = factory.Sequence(lambda n: f"vector-{n}")

    class Meta:
        model = _Vector


class _VectorEmbedding(vector_storage.VectorEmbedding):
    name: str

    class _Config(vector_storage.Config):
        collection = vector_storage.Collection.TESTING_VECTOR_EMBEDDINGS
        category = vector_storage.DataCategory.TESTING


class VectorEmbedding(_base.Factory):
    name = factory.Sequence(lambda n: f"vector-{n}")
    embedding = factory.LazyFunction(lambda: [1.0] + [0.0] * 383)

    class Meta:
        model = _VectorEmbedding


# Cleaning.


class _CleanedDocument(_base.Factory):
    raw_document_id = factory.Sequence(lambda n: f"raw-document-{n}")
    content = factory.Sequence(lambda n: f"content-{n}")
    platform = "some-platform"

    author = factory.SubFactory(documents.Author)
    author_id = factory.LazyAttribute(lambda o: o.author.id)
    author_full_name = factory.LazyAttribute(lambda o: o.author.full_name)


class CleanedArticle(_CleanedDocument, _mixins.ArticleMixin):
    class Meta:
        model = cleaning.CleanedArticle


class CleanedRepository(_CleanedDocument, _mixins.RepositoryMixin):
    class Meta:
        model = cleaning.CleanedRepository


# Chunking.


class _Chunk(_CleanedDocument):
    cleaned_document_id = factory.Sequence(lambda n: f"cleaned-document-{n}")


class ArticleChunk(_Chunk, _mixins.ArticleMixin):
    class Meta:
        model = chunking.ArticleChunk


class RepositoryChunk(_Chunk, _mixins.RepositoryMixin):
    class Meta:
        model = chunking.RepositoryChunk


# Embedding.


class _Embedding(_Chunk):
    embedding = factory.LazyFunction(lambda: [1.0] + [0.0] * 383)


class EmbeddedArticleChunk(_Embedding, _mixins.ArticleMixin):
    class Meta:
        model = embedding.EmbeddedArticleChunk


class EmbeddedRepositoryChunk(_Embedding, _mixins.RepositoryMixin):
    class Meta:
        model = embedding.EmbeddedRepositoryChunk
