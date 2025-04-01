from __future__ import annotations

import dataclasses

import loguru
from pymongo import collection as pymongo_collection
from pymongo import database as pymongo_database
from pymongo import errors as pymongo_errors
from pymongo import mongo_client

from llm_twin.domain import raw_documents


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
        self, collection: raw_documents.Collection
    ) -> pymongo_collection.Collection:
        assert self._database is not None  # For mypy.
        return self._database[collection.value]


@dataclasses.dataclass
class MongoDatabase(raw_documents.NoSQLDatabase):
    _connector: MongoDatabaseConnector

    def find_one(
        self, *, collection: raw_documents.Collection, **filter_options: object
    ) -> raw_documents.RawDocument:
        mongo_collection = self._connector.get_collection(collection)

        if (result := mongo_collection.find_one(filter_options)) is None:
            raise raw_documents.DocumentDoesNotExist
        return result

    def insert_one(
        self,
        *,
        collection: raw_documents.Collection,
        document: raw_documents.RawDocument,
    ) -> None:
        mongo_collection = self._connector.get_collection(collection)

        try:
            mongo_collection.insert_one(document)
        except pymongo_errors.WriteError as exc:
            raise raw_documents.UnableToSaveDocument from exc
