from __future__ import annotations

import dataclasses
import typing

import loguru
from pymongo import collection as pymongo_collection
from pymongo import database as pymongo_database
from pymongo import errors as pymongo_errors
from pymongo import mongo_client

from llm_twin.domain.storage import document as document_storage


class MongoDatabaseConnector:
    _client: mongo_client.MongoClient
    _database: pymongo_database.Database

    def __new__(
        cls,
        database_host: str,
        database_name: str,
        *args: object,
        **kwargs: object,
    ) -> MongoDatabaseConnector:
        if not hasattr(cls, "_client"):
            try:
                cls._client = mongo_client.MongoClient(database_host)
            except pymongo_errors.ConnectionFailure as exc:
                loguru.logger.error(f"Couldn't connect to the database: {str(exc)}")
                raise

        if not hasattr(cls, "_database"):
            cls._database = cls._client.get_database(database_name)

        loguru.logger.info(
            f"Connection to MongoDB with URI successful: {database_host}"
        )

        return super().__new__(cls)

    def get_collection(
        self, document_class: type[document_storage.Document]
    ) -> pymongo_collection.Collection:
        collection_name = document_class.get_collection_name()
        assert self._database is not None  # For mypy.
        return self._database[collection_name.value]


@dataclasses.dataclass
class MongoDatabase(document_storage.DocumentDatabase):
    _connector: MongoDatabaseConnector

    def find_one(
        self,
        *,
        document_class: type[document_storage.DocumentT],
        **filter_options: object,
    ) -> document_storage.DocumentT:
        mongo_collection = self._connector.get_collection(document_class)

        if (serialized_document := mongo_collection.find_one(filter_options)) is None:
            raise document_storage.DocumentDoesNotExist
        return _deserialize(
            serialized_document=serialized_document, document_class=document_class
        )

    def find_many(
        self,
        *,
        document_class: type[document_storage.DocumentT],
        **filter_options: object,
    ) -> list[document_storage.DocumentT]:
        mongo_collection = self._connector.get_collection(document_class)
        serialized_documents = mongo_collection.find(filter_options)

        return [
            _deserialize(
                serialized_document=serialized_document, document_class=document_class
            )
            for serialized_document in serialized_documents
        ]

    def insert_one(self, *, document: document_storage.Document) -> None:
        mongo_collection = self._connector.get_collection(type(document))
        serialized_document = _serialize(document=document)

        try:
            mongo_collection.insert_one(serialized_document)
        except pymongo_errors.WriteError as exc:
            raise document_storage.UnableToSaveDocument from exc


# Serialization.


def _deserialize(
    *,
    serialized_document: dict[str, typing.Any],
    document_class: type[document_storage.DocumentT],
) -> document_storage.DocumentT:
    if not serialized_document:
        raise document_storage.DocumentIsEmpty()

    return document_class(**serialized_document)


def _serialize(*, document: document_storage.Document) -> dict[str, typing.Any]:
    return document.model_dump(by_alias=True)
