import abc
import enum
import typing

import pydantic

from llm_twin.domain.storage import _ids


class Collection(enum.Enum):
    CLEANED_ARTICLES = "cleaned_articles"
    CLEANED_REPOSITORIES = "cleaned_repositories"

    EMBEDDED_ARTICLES = "embedded_articles"
    EMBEDDED_REPOSITORIES = "embedded_repositories"

    # Testing.
    TESTING_VECTORS = "testing_vectors"
    TESTING_VECTOR_EMBEDDINGS = "testing_vector_embeddings"


class DataCategory(enum.Enum):
    POSTS = "posts"
    ARTICLES = "articles"
    REPOSITORIES = "repositories"
    TESTING = "testing"


class Config(pydantic.BaseModel):
    collection: typing.ClassVar[Collection]
    category: typing.ClassVar[DataCategory]


class Vector(pydantic.BaseModel, abc.ABC):
    id: str = pydantic.Field(default_factory=_ids.generate_id)

    _Config: typing.ClassVar[type[Config]]

    @classmethod
    def collection(cls) -> Collection:
        return cls._Config.collection

    @classmethod
    def category(cls) -> DataCategory:
        return cls._Config.category


class VectorEmbedding(Vector, abc.ABC):
    embedding: list[float]
