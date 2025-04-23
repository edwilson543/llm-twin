from __future__ import annotations

import abc
import enum

import pydantic

from llm_twin.domain.storage import _ids


class Collection(enum.StrEnum):
    ARTICLES = "articles"
    POSTS = "posts"
    REPOSITORIES = "repositories"
    AUTHORS = "authors"


class Document(pydantic.BaseModel, abc.ABC):
    id: str = pydantic.Field(default_factory=_ids.generate_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    @abc.abstractmethod
    def get_collection_name(cls) -> Collection:
        """
        Get the name of the collection used to store this document type in the database.
        """
        raise NotImplementedError
