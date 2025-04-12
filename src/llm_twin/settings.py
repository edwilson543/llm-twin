from __future__ import annotations

import pydantic_settings

from llm_twin.domain import models
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

    # Embeddings.
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_CACHE_DIR: str | None = None

    @classmethod
    def load_settings(cls) -> Settings:
        return cls()


settings = Settings.load_settings()

# Databases.


def get_document_database() -> document_storage.DocumentDatabase:
    connector = mongo.MongoDatabaseConnector(
        database_host=settings.MONGO_DATABASE_HOST,
        database_name=settings.MONGO_DATABASE_NAME,
    )
    return mongo.MongoDatabase(_connector=connector)


def get_vector_database() -> vector_storage.VectorDatabase:
    embedding_model_config = _get_embedding_model_config()
    return qdrant.QdrantDatabase.build(
        host=settings.QDRANT_DATABASE_HOST,
        port=settings.QDRANT_DATABASE_PORT,
        embedding_model_config=embedding_model_config,
    )


# Models.


def _get_embedding_model_config() -> models.EmbeddingModelConfig:
    configs: dict[models.EmbeddingModelName, models.EmbeddingModelConfig] = {
        models.EmbeddingModelName.MINILM: models.EmbeddingModelConfig(
            model_name=models.EmbeddingModelName.MINILM,
            embedding_size=384,
            max_input_length=256,
            cache_dir=settings.MODEL_CACHE_DIR,
        )
    }

    model_name = models.EmbeddingModelName(settings.EMBEDDING_MODEL_NAME)
    return configs[model_name]


def get_embedding_model() -> models.EmbeddingModel:
    config = _get_embedding_model_config()
    return models.SentenceTransformerEmbeddingModel(config=config)
