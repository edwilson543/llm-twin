from llm_twin.domain import models
from llm_twin.domain.feature_engineering import embedding


def rerank_documents(
    *,
    query: str,
    documents: list[embedding.EmbeddedChunk],
    top_k: int,
    cross_encoder_model: models.CrossEncoderModel,
) -> list[embedding.EmbeddedChunk]:
    """
    Return the `top_k` documents that are most relevant to the query.
    """
    if len(documents) <= top_k:
        return documents

    pairs = [(query, document.to_context()) for document in documents]
    predictions = cross_encoder_model.predict(pairs=pairs)

    ranked = [
        chunk
        for chunk, _ in sorted(zip(documents, predictions), key=lambda pair: pair[1])
    ]

    return ranked[:top_k]
