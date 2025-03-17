from __future__ import annotations

import loguru
from pymongo import collection as pymongo_collection
from pymongo import database as pymongo_database
from pymongo import errors as pymongo_errors
from pymongo import mongo_client

from llm_twin.domain import documents
from llm_twin.settings import settings


class _MongoDatabaseConnector:
    _client: mongo_client.MongoClient | None = None
    _database: pymongo_database.Database | None = None

    def __new__(cls, *args: object, **kwargs: object) -> _MongoDatabaseConnector:
        if cls._client is None:
            try:
                cls._client = mongo_client.MongoClient(settings.MONGO_DATABASE_HOST)
            except pymongo_errors.ConnectionFailure as exc:
                loguru.logger.error(f"Couldn't connect to the database: {str(exc)}")
                raise

        if cls._database is None:
            cls._database = cls._client.get_database(settings.MONGO_DATABASE_NAME)

        loguru.logger.info(
            f"Connection to MongoDB with URI successful: {settings.MONGO_DATABASE_HOST}"
        )

        return cls(*args, **kwargs)

    def get_collection(self, collection: str) -> pymongo_collection.Collection:
        assert self._database is not None  # For mypy.
        return self._database[collection]


_mongo_database = _MongoDatabaseConnector()


class MongoDatabase(documents.NoSQLDatabase):
    def find_one(
        self, *, collection: str, **filter_options: object
    ) -> documents.RawDocument | None:
        mongo_collection = _mongo_database.get_collection(collection)
        return mongo_collection.find_one(filter_options)

    def insert_one(self, *, collection: str, document: documents.RawDocument) -> None:
        mongo_collection = _mongo_database.get_collection(collection)

        try:
            mongo_collection.insert_one(document)
        except pymongo_errors.WriteError as exc:
            raise documents.UnableToSaveDocument from exc
