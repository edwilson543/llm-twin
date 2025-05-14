from __future__ import annotations

import pathlib

import pydantic_settings

from llm_twin.domain import models


DOTENV_FILE = pathlib.Path(__file__).parents[3] / ".env"


class Settings(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        env_file=str(DOTENV_FILE), env_file_encoding="utf-8", extra="ignore"
    )

    # Mongo.
    MONGO_DATABASE_HOST: str = "mongodb://mongo_user:mongo_password@127.0.0.1:27017"
    MONGO_DATABASE_NAME: str = "llm-twin"

    # Qdrant.
    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333

    # Embeddings.
    EMBEDDING_MODEL_NAME: models.EmbeddingModelName = models.EmbeddingModelName.MINILM
    EMBEDDING_MODEL_MAX_INPUT_LENGTH: int = 256
    EMBEDDING_MODEL_CACHE_DIR: str | None = None
    EMBEDDING_SIZE: int = 384

    # Retrieval.
    RETRIEVAL_CROSS_ENCODER_MODEL_NAME: models.CrossEncoderModelName = (
        models.CrossEncoderModelName.TINY_BERT
    )
    RETRIEVAL_NUMBER_OF_QUERY_EXPANSIONS: int = 3
    RETRIEVAL_MAX_DOCUMENTS_PER_QUERY: int = 3

    # Training.
    TRAINING_OUTPUT_DIR: str = "/tmp/llm-twin/"

    # Vendors.
    AWS_SAGEMAKER_ROLE_ARN: str = ""

    COMET_API_KEY: str = ""
    COMET_PROJECT_NAME: str = "llm-twin"

    HUGGINGFACE_ACCESS_TOKEN: str = ""
    HUGGINGFACE_DATASET_WORKSPACE: str = ""

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL_TAG: str = "gpt-4o-mini"

    @classmethod
    def load_settings(cls) -> Settings:
        return cls()


settings = Settings.load_settings()
