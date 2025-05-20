import pathlib

from llm_twin.domain import inference, models, rag
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
        api_key=settings.QDRANT_DATABASE_API_KEY,
        embedding_model_config=embedding_model_config,
    )


# Models.


def _get_embedding_model_config() -> models.EmbeddingModelConfig:
    configs: dict[models.EmbeddingModelName, models.EmbeddingModelConfig] = {
        models.EmbeddingModelName.MINILM: models.EmbeddingModelConfig(
            model_name=settings.EMBEDDING_MODEL_NAME,
            embedding_size=settings.EMBEDDING_SIZE,
            max_input_length=settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH,
            cache_dir=settings.EMBEDDING_MODEL_CACHE_DIR,
        )
    }

    return configs[settings.EMBEDDING_MODEL_NAME]


def get_embedding_model() -> models.EmbeddingModel:
    config = _get_embedding_model_config()
    return models.SentenceTransformerEmbeddingModel(config=config)


def get_language_model() -> models.LanguageModel:
    if not (settings.OPENAI_API_KEY and settings.OPENAI_MODEL_TAG):
        raise ConfigurationError("Language model settings are not set.")

    return models.OpenAILanguageModel(
        api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_MODEL_TAG
    )


def get_cross_encoder_model() -> models.CrossEncoderModel:
    return models.SentenceTransformerCrossEncoder(
        model_name=settings.RETRIEVAL_CROSS_ENCODER_MODEL_NAME
    )


# RAG.


def get_retrieval_config() -> rag.RetrievalConfig:
    return rag.RetrievalConfig(
        db=get_vector_database(),
        language_model=get_language_model(),
        embedding_model=get_embedding_model(),
        cross_encoder_model=get_cross_encoder_model(),
        max_documents_per_query=settings.RETRIEVAL_MAX_DOCUMENTS_PER_QUERY,
        number_of_query_expansions=settings.RETRIEVAL_NUMBER_OF_QUERY_EXPANSIONS,
    )


# Training.


def get_training_output_dir() -> pathlib.Path:
    return pathlib.Path(settings.TRAINING_OUTPUT_DIR)


def login_to_comet_ml() -> None:
    import comet_ml

    comet_ml.login(
        api_key=settings.COMET_API_KEY, project_name=settings.COMET_PROJECT_NAME
    )


def get_inference_engine() -> inference.InferenceEngine:
    load_model_from = get_training_output_dir() / "dpo"
    return inference.LocalInferenceEngine(load_model_from=load_model_from)


# def get_training_runner() -> training.Runner:
#     return training.SageMaker(
#         _aws_role_arn=settings.AWS_SAGEMAKER_ROLE_ARN,
#         _comet_api_key=settings.COMET_API_KEY,
#         _comet_project_name=settings.COMET_PROJECT_NAME,
#         _hugging_face_access_token=settings.HUGGINGFACE_ACCESS_TOKEN,
#         _huggingface_dataset_workspace=settings.HUGGINGFACE_DATASET_WORKSPACE,
#     )
