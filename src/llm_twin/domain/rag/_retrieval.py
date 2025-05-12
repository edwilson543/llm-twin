from llm_twin.domain.feature_engineering import embedding
from llm_twin.domain.storage import vector as vector_storage

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

    all_chunks = [
        _search_vector_db(
            query=query, max_chunks=config.max_chunks_per_query, db=config.db
        )
        for query in expansion.all_queries
    ]
    unique_chunks = _get_unique_chunks(chunks=all_chunks)

    return _reranking.filter_most_relevant_context(
        query=query,
        chunks=unique_chunks,
        max_chunks=config.max_chunks_per_query,
        cross_encoder_model=config.cross_encoder_model,
    )


def _search_vector_db(
    *,
    query: str,
    max_chunks: int,
    db: vector_storage.VectorDatabase,
) -> list[embedding.EmbeddedChunk]:
    """
    Search the vector database for the context chunks most relevant to the query.
    """
    # TODO -> implement.
    raise NotImplementedError


def _get_unique_chunks(
    *,
    chunks: list[list[embedding.EmbeddedChunk]],
) -> list[embedding.EmbeddedChunk]:
    raise NotImplementedError
