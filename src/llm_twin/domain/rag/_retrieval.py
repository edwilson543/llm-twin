from llm_twin.domain.feature_engineering import embedding

from . import _config, _query_expansion, _reranking


def retrieve_context_for_query(
    *, query: str, config: _config.RAGConfig
) -> list[embedding.EmbeddedChunk]:
    """
    Augment a query with context retrieved from the database.
    """
    expansion = _query_expansion.expand_query(
        query=query,
        number_of_query_expansions=config.number_of_query_expansions,
        language_model=config.language_model,
    )

    query_vectors = config.embedding_model.generate_embeddings(
        input_text=expansion.all_queries
    )
    query_classes: list[type[embedding.EmbeddedChunk]] = [
        embedding.EmbeddedArticleChunk,
        embedding.EmbeddedRepositoryChunk,
    ]

    all_chunks = [
        config.db.vector_search(
            vector_class=klass,
            query_vector=query_vector,
            limit=config.max_chunks_per_query,
        )
        for query_vector in query_vectors
        for klass in query_classes
    ]
    unique_chunks = _get_unique_chunks(chunks=all_chunks)

    return _reranking.filter_most_relevant_context(
        query=query,
        chunks=unique_chunks,
        max_chunks=config.max_chunks_per_query,
        cross_encoder_model=config.cross_encoder_model,
    )


def _get_unique_chunks(
    *,
    chunks: list[list[embedding.EmbeddedChunk]],
) -> list[embedding.EmbeddedChunk]:
    unique_chunks: dict[str, embedding.EmbeddedChunk] = {}

    for chunk_list in chunks:
        for chunk in chunk_list:
            if chunk.id not in unique_chunks.keys():
                unique_chunks[chunk.id] = chunk

    return list(unique_chunks.values())
