from __future__ import annotations

import abc
import dataclasses
import enum
import typing


SerializedRawDocument = dict[str, typing.Any]


@dataclasses.dataclass(frozen=True)
class DocumentDoesNotExist(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class UnableToSaveDocument(Exception):
    pass


class Collection(enum.Enum):
    ARTICLES = "articles"
    POSTS = "posts"
    REPOSITORIES = "repositories"
    AUTHORS = "authors"


class RawDocumentDatabase(abc.ABC):
    @abc.abstractmethod
    def find_one(
        self, *, collection: Collection, **filter_options: object
    ) -> SerializedRawDocument:
        """
        Find a document in a collection, using some filters.

        :raises DocumentDoesNotExist: If no document was found.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def find_many(
        self, *, collection: Collection, **filter_options: object
    ) -> list[SerializedRawDocument]:
        """
        Find all matching documents in a collection, using some filters.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def insert_one(
        self, *, collection: Collection, document: SerializedRawDocument
    ) -> None:
        """
        Save a document in a collection.

        :raises UnableToSaveDocument: If the operation fails for some reason.
        """
        raise NotImplementedError
