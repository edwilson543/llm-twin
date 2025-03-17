from __future__ import annotations

import abc
import dataclasses
import typing


RawDocument = dict[str, typing.Any]


@dataclasses.dataclass(frozen=True)
class DocumentDoesNotExist(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class UnableToSaveDocument(Exception):
    pass


class NoSQLDatabase(abc.ABC):
    @abc.abstractmethod
    def find_one(self, *, collection: str, **filter_options: object) -> RawDocument:
        """
        Find a document in a collection, using some filters.

        :raises DocumentDoesNotExist: If no document was found.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def insert_one(self, *, collection: str, document: RawDocument) -> None:
        """
        Save a document in a collection.

        :raises UnableToSaveDocument: If the operation fails for some reason.
        """
        raise NotImplementedError
