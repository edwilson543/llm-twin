from __future__ import annotations

import abc
import dataclasses
import typing
import uuid

import pydantic
from pymongo import errors as pymongo_errors

from . import _mongodb


DocumentType = typing.TypeVar("DocumentType", bound="BaseDocument")


@dataclasses.dataclass(frozen=True)
class DocumentIsEmpty(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class UnableToSaveDocument(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class UnableToCreateDocument(Exception):
    pass


class NoSQLBaseDocument(pydantic.BaseModel, typing.Generic[DocumentType], abc.ABC):
    id: pydantic.UUID4 = pydantic.Field(default_factory=uuid.uuid4)

    def __eq__(self, other: NoSQLBaseDocument) -> bool:
        if not isinstance(other, self.__class__):
            return False

        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    # Database operations.

    @classmethod
    def get_or_create(cls, **filter_options: object) -> DocumentType:
        collection = _mongodb.database[cls._get_collection_name()]

        if raw_data := collection.find_one(filter_options):
            instance = cls.deserialize(raw_data=raw_data)
        else:
            instance = cls(**filter_options)
            try:
                instance.save()
            except pymongo_errors.OperationFailure as exc:
                raise UnableToCreateDocument from exc
        return instance

    def save(self, exclude_unset: bool = False, by_alias: bool = True) -> None:
        """
        Insert this document into the relevant MongoDB collection.
        """
        collection = _mongodb.database[self._get_collection_name()]
        serialized = self.serialize(exclude_unset=exclude_unset, by_alias=by_alias)

        try:
            collection.insert_one(serialized)
        except pymongo_errors.WriteError as exc:
            raise UnableToSaveDocument from exc

    # Serialization.

    @classmethod
    def deserialize(cls, raw_data: dict) -> DocumentType:
        """
        Deserialize a document using the raw data retrieved from the database.
        """
        if not raw_data:
            raise DocumentIsEmpty()

        id = raw_data.pop("_id")
        return cls(id=id, **raw_data)

    def serialize(
        self, exclude_unset: bool = False, by_alias: bool = True
    ) -> dict[str, typing.Any]:
        """
        Serialize a document into raw database for persistence in the database.
        """
        parsed = self.model_dump(exclude_unset=exclude_unset, by_alias=by_alias)
        if "_id" not in parsed and "id" in parsed:
            parsed["_id"] = str(parsed.pop("id"))

        for key, value in parsed.items():
            if isinstance(value, uuid.UUID):
                parsed[key] = str(value)

        return parsed

    @classmethod
    @abc.abstractmethod
    def _get_collection_name(cls) -> str:
        """
        Get the name of the collection used to store this document type in the database.
        """
        raise NotImplementedError
