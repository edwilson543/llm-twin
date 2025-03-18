from __future__ import annotations

import dataclasses

import loguru
from pymongo import collection as pymongo_collection
from pymongo import database as pymongo_database
from pymongo import errors as pymongo_errors
from pymongo import mongo_client

from llm_twin import settings
from llm_twin.domain import documents


class MongoDatabaseConnector:
    _client: mongo_client.MongoClient
    _database: pymongo_database.Database

    def __new__(
        cls,
        settings: settings.Settings = settings.settings,
        *args: object,
        **kwargs: object,
    ) -> MongoDatabaseConnector:
        if not hasattr(cls, "_client"):
            try:
                cls._client = mongo_client.MongoClient(settings.MONGO_DATABASE_HOST)
            except pymongo_errors.ConnectionFailure as exc:
                loguru.logger.error(f"Couldn't connect to the database: {str(exc)}")
                raise

        if not hasattr(cls, "_database"):
            cls._database = cls._client.get_database(settings.MONGO_DATABASE_NAME)

        loguru.logger.info(
            f"Connection to MongoDB with URI successful: {settings.MONGO_DATABASE_HOST}"
        )

        return super().__new__(cls)

    def get_collection(self, collection: str) -> pymongo_collection.Collection:
        assert self._database is not None  # For mypy.
        return self._database[collection]


@dataclasses.dataclass
class MongoDatabase(documents.NoSQLDatabase):
    _connector: MongoDatabaseConnector = dataclasses.field(
        default_factory=MongoDatabaseConnector
    )

    def find_one(
        self, *, collection: str, **filter_options: object
    ) -> documents.RawDocument:
        mongo_collection = self._connector.get_collection(collection)

        if (result := mongo_collection.find_one(filter_options)) is None:
            raise documents.DocumentDoesNotExist
        return result

    def insert_one(self, *, collection: str, document: documents.RawDocument) -> None:
        mongo_collection = self._connector.get_collection(collection)

        try:
            mongo_collection.insert_one(document)
        except pymongo_errors.WriteError as exc:
            raise documents.UnableToSaveDocument from exc
