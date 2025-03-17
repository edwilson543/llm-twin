import loguru
from pymongo import MongoClient, mongo_client
from pymongo import errors as pymongo_errors

from llm_twin.settings import settings


class MongoDatabaseConnector:
    _client: MongoClient | None = None

    def __new__(cls, *args, **kwargs) -> MongoClient:
        if cls._client is None:
            try:
                cls._client = mongo_client.MongoClient(settings.DATABASE_HOST)
            except pymongo_errors.ConnectionFailure as exc:
                loguru.logger.error(f"Couldn't connect to the database: {str(exc)}")
                raise

        loguru.logger.info(
            f"Connection to MongoDB with URI successful: {settings.DATABASE_HOST}"
        )

        return cls._client


_connection = MongoDatabaseConnector()
database = _connection.get_database(settings.MONGO_DATABASE_NAME)
