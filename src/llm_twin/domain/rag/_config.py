import dataclasses

from llm_twin.domain import models
from llm_twin.domain.storage import vector as vector_storage


@dataclasses.dataclass(frozen=True)
class RAGConfig:
    # Databases.
    db: vector_storage.VectorDatabase

    # Third party models.
    language_model: models.LanguageModel
    embedding_model: models.EmbeddingModel
    cross_encoder_model: models.CrossEncoderModel

    # Parameters.
    number_of_query_expansions: int
    max_chunks_per_query: int
