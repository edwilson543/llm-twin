from __future__ import annotations

import dataclasses

import loguru
from pymongo import database as pymongo_database
from pymongo import errors as pymongo_errors
from pymongo import mongo_client

from llm_twin import settings
from llm_twin.domain import documents


class _MongoDatabaseConnector:
    _client: mongo_client.MongoClient | None = None
    _database: pymongo_database.Database | None = None

    def __new__(
        cls,
        settings: settings.Settings = settings.settings,
        *args: object,
        **kwargs: object,
    ) -> pymongo_database.Database:
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

        return cls._database


@dataclasses.dataclass
class MongoDatabase(documents.NoSQLDatabase):
    _db: pymongo_database.Database = dataclasses.field(
        default_factory=_MongoDatabaseConnector
    )

    def find_one(
        self, *, collection: str, **filter_options: object
    ) -> documents.RawDocument:
        mongo_collection = self._db[collection]
        if (result := mongo_collection.find_one(filter_options)) is None:
            raise documents.DocumentDoesNotExist
        return result

    def insert_one(self, *, collection: str, document: documents.RawDocument) -> None:
        mongo_collection = self._db[collection]

        try:
            mongo_collection.insert_one(document)
        except pymongo_errors.WriteError as exc:
            raise documents.UnableToSaveDocument from exc
