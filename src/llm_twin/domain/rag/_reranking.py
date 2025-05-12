from llm_twin.domain import models
from llm_twin.domain.feature_engineering import embedding


def filter_most_relevant_context(
    *,
    query: str,
    chunks: list[embedding.EmbeddedChunk],
    max_chunks: int,
    cross_encoder_model: models.CrossEncoderModel,
) -> list[embedding.EmbeddedChunk]:
    """
    Use the cross encoder to filter out the context chunks least relevant to the query.
    """
    if len(chunks) <= max_chunks:
        return chunks

    raise NotImplementedError
