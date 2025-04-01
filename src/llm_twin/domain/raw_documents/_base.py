from __future__ import annotations

import abc
import dataclasses
import typing
import uuid

import pydantic

from . import _db


@dataclasses.dataclass(frozen=True)
class DocumentIsEmpty(Exception):
    pass


class RawDocument(pydantic.BaseModel, abc.ABC):
    id: pydantic.UUID4 = pydantic.Field(default_factory=uuid.uuid4)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    # Database operations.

    @classmethod
    def get(
        cls, *, db: _db.RawDocumentDatabase, **filter_options: typing.Any
    ) -> typing.Self:
        """
        Find a document from the database.

        :raises DocumentDoesNotExist: If no such document was found.
        """
        collection = cls.get_collection_name()
        serialized_document = db.find_one(collection=collection, **filter_options)
        return cls.deserialize(serialized_document)

    @classmethod
    def filter(
        cls, *, db: _db.RawDocumentDatabase, **filter_options: typing.Any
    ) -> list[typing.Self]:
        """
        Find all matching documents from the database.

        :raises DocumentDoesNotExist: If no such document was found.
        """
        collection = cls.get_collection_name()
        serialized_documents = db.find_many(collection=collection, **filter_options)
        return [cls.deserialize(document) for document in serialized_documents]

    @classmethod
    def get_or_create(
        cls, *, db: _db.RawDocumentDatabase, **filter_options: typing.Any
    ) -> typing.Self:
        """
        Get or create a document from the database.

        :raises UnableToSaveDocument if the document had to be created but could
            not be saved in the database.
        """
        try:
            instance = cls.get(db=db, filter_options=filter_options)
        except _db.DocumentDoesNotExist:
            instance = cls(**filter_options)
            instance.save(db=db)

        return instance

    def save(self, *, db: _db.RawDocumentDatabase) -> None:
        """
        Insert this document into the relevant MongoDB collection.

        :raises UnableToSaveDocument: If the operation fails for some reason.
        """
        collection = self.get_collection_name()
        serialized = self.serialize()
        db.insert_one(collection=collection, document=serialized)

    # Serialization.

    @classmethod
    def deserialize(cls, raw_document: dict) -> typing.Self:
        """
        Deserialize a document using the raw data retrieved from the database.
        """
        if not raw_document:
            raise DocumentIsEmpty()

        id = raw_document.pop("_id")
        return cls(id=id, **raw_document)

    def serialize(self) -> dict[str, typing.Any]:
        """
        Serialize a document into raw database for persistence in the database.
        """
        parsed = self.model_dump()
        if "_id" not in parsed and "id" in parsed:
            parsed["_id"] = str(parsed.pop("id"))

        for key, value in parsed.items():
            if isinstance(value, uuid.UUID):
                parsed[key] = str(value)

        return parsed

    @classmethod
    @abc.abstractmethod
    def get_collection_name(cls) -> _db.Collection:
        """
        Get the name of the collection used to store this document type in the database.
        """
        raise NotImplementedError


class ExtractedDocument(RawDocument, abc.ABC):
    """
    A document that was extracted from some webpage, by a crawler.
    """

    content: dict
    platform: str
    author_id: pydantic.UUID4
    author_full_name: str
