from __future__ import annotations

import pydantic_settings

from llm_twin.domain.storage import document as document_storage
from llm_twin.infrastructure.db import mongo


class Settings(pydantic_settings.BaseSettings):
    # Raw document database.
    MONGO_DATABASE_HOST: str = "mongodb://mongo_user:mongo_password@127.0.0.1:27017"
    MONGO_DATABASE_NAME: str = "llm-twin"

    @classmethod
    def load_settings(cls) -> Settings:
        return cls()


settings = Settings.load_settings()


def get_raw_document_database() -> document_storage.RawDocumentDatabase:
    connector = mongo.MongoDatabaseConnector(
        database_host=settings.MONGO_DATABASE_HOST,
        database_name=settings.MONGO_DATABASE_NAME,
    )
    return mongo.MongoDatabase(_connector=connector)
