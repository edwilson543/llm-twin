from __future__ import annotations

import pydantic_settings

from llm_twin.domain.storage import document as document_storage
from llm_twin.domain.storage import vector as vector_storage
from llm_twin.infrastructure.db import mongo, qdrant


class Settings(pydantic_settings.BaseSettings):
    # Mongo.
    MONGO_DATABASE_HOST: str = "mongodb://mongo_user:mongo_password@127.0.0.1:27017"
    MONGO_DATABASE_NAME: str = "llm-twin"

    # Qdrant.
    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333

    @classmethod
    def load_settings(cls) -> Settings:
        return cls()


settings = Settings.load_settings()


def get_document_database() -> document_storage.DocumentDatabase:
    connector = mongo.MongoDatabaseConnector(
        database_host=settings.MONGO_DATABASE_HOST,
        database_name=settings.MONGO_DATABASE_NAME,
    )
    return mongo.MongoDatabase(_connector=connector)


def get_vector_database() -> vector_storage.VectorDatabase:
    connector = qdrant.QdrantDatabaseConnector(
        database_host=settings.QDRANT_DATABASE_HOST,
        database_port=settings.QDRANT_DATABASE_PORT,
    )
    return qdrant.QdrantDatabase(_connector=connector)
