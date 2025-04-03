import abc
import enum
import typing
import uuid

import pydantic


def generate_id() -> str:
    return str(uuid.uuid4())


class Collection(enum.Enum):
    CLEANED_ARTICLES = "articles"
    CLEANED_REPOSITORIES = "cleaned_repositories"

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
    id: str = pydantic.Field(default_factory=generate_id)

    _Config: typing.ClassVar[type[Config]]

    @classmethod
    def collection(cls) -> Collection:
        return cls._Config.collection


class VectorEmbedding(Vector, abc.ABC):
    embedding: list[float]
