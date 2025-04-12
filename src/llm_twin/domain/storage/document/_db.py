from __future__ import annotations

import abc
import dataclasses
import typing

from . import _document


@dataclasses.dataclass(frozen=True)
class DocumentDoesNotExist(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class UnableToSaveDocument(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class DocumentIsEmpty(Exception):
    pass


DocumentT = typing.TypeVar("DocumentT", bound=_document.Document)


class DocumentDatabase(abc.ABC):
    def get_or_create(
        self, *, document_class: type[DocumentT], **filter_options: typing.Any
    ) -> DocumentT:
        try:
            document = self.find_one(document_class=document_class, **filter_options)
        except DocumentDoesNotExist:
            document = document_class(**filter_options)
            self.insert_one(document=document)
        return document

    @abc.abstractmethod
    def find_one(
        self, *, document_class: type[DocumentT], **filter_options: object
    ) -> DocumentT:
        """
        Find a document in a collection, using some filters.

        :raises DocumentDoesNotExist: If no document was found.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def find_many(
        self, *, document_class: type[DocumentT], **filter_options: object
    ) -> list[DocumentT]:
        """
        Find all matching documents in a collection, using some filters.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def insert_one(self, *, document: _document.Document) -> None:
        """
        Save a document in a collection.

        :raises UnableToSaveDocument: If the operation fails for some reason.
        """
        raise NotImplementedError
