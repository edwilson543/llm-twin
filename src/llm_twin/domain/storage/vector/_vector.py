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


class DataCategory(enum.Enum):
    POSTS = "posts"
    ARTICLES = "articles"
    REPOSITORIES = "repositories"


class Config(pydantic.BaseModel):
    collection: typing.ClassVar[Collection]
    category: typing.ClassVar[DataCategory]
    use_vector_index: typing.ClassVar[bool]


class Vector(pydantic.BaseModel, abc.ABC):
    id: str = pydantic.Field(default_factory=generate_id)

    _Config: typing.ClassVar[type[Config]]

    @property
    def collection(self) -> Collection:
        return self._Config.collection
