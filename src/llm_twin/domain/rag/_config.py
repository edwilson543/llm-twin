import pydantic

from llm_twin.domain import models
from llm_twin.domain.storage import vector as vector_storage


class RAGConfig(pydantic.BaseModel):
    # Databases.
    db: vector_storage.VectorDatabase

    # Third party models.
    language_model: models.LanguageModel
    cross_encoder_model: models.CrossEncoderModel

    # Parameters.
    number_of_query_expansions: int
    max_chunks_per_query: int
