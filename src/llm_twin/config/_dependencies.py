from llm_twin.domain import models
from llm_twin.domain.storage import document as document_storage
from llm_twin.domain.storage import vector as vector_storage
from llm_twin.infrastructure.db import mongo, qdrant

from ._settings import settings


class ConfigurationError(Exception):
    pass


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


def get_language_model() -> models.LanguageModel:
    if not (settings.OPENAI_API_KEY and settings.OPENAI_MODEL_TAG):
        raise ConfigurationError("Language model settings are not set.")

    return models.OpenAILanguageModel(
        api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_MODEL_TAG
    )


# Training.


# def get_training_runner() -> training.Runner:
#     return training.SageMaker(
#         _aws_role_arn=settings.AWS_SAGEMAKER_ROLE_ARN,
#         _comet_api_key=settings.COMET_API_KEY,
#         _comet_project_name=settings.COMET_PROJECT_NAME,
#         _hugging_face_access_token=settings.HUGGINGFACE_ACCESS_TOKEN,
#         _huggingface_dataset_workspace=settings.HUGGINGFACE_DATASET_WORKSPACE,
#     )
