import dataclasses

from llm_twin.domain import models
from llm_twin.domain.feature_engineering import embedding
from llm_twin.domain.rag._retrieval import _query_expansion, _reranking
from llm_twin.domain.storage import vector as vector_storage


@dataclasses.dataclass(frozen=True)
class RetrievalConfig:
    # Databases.
    db: vector_storage.VectorDatabase

    # Third party models.
    language_model: models.LanguageModel
    embedding_model: models.EmbeddingModel
    cross_encoder_model: models.CrossEncoderModel

    # Parameters.
    number_of_query_expansions: int
    max_documents_per_query: int


def retrieve_relevant_documents(
    *, query: str, config: RetrievalConfig
) -> list[embedding.EmbeddedChunk]:
    """
    Augment a query with context retrieved from the database.
    """
    expansion = _query_expansion.Expansion.from_query(
        query=query,
        n_expansions=config.number_of_query_expansions,
        language_model=config.language_model,
    )

    query_vectors = config.embedding_model.generate_embeddings(
        input_text=expansion.all_queries
    )

    all_documents = _search_vector_db(
        query_vectors=query_vectors,
        db=config.db,
        max_documents_per_query=config.max_documents_per_query,
    )
    unique_documents = _get_unique_documents(documents=all_documents)

    return _reranking.rerank_documents(
        query=query,
        documents=unique_documents,
        top_k=config.max_documents_per_query,
        cross_encoder_model=config.cross_encoder_model,
    )


def _search_vector_db(
    *,
    query_vectors: list[list[float]],
    db: vector_storage.VectorDatabase,
    max_documents_per_query: int,
) -> list[list[embedding.EmbeddedChunk]]:
    query_classes: list[type[embedding.EmbeddedChunk]] = [
        embedding.EmbeddedArticleChunk,
        embedding.EmbeddedRepositoryChunk,
    ]

    return [
        db.vector_search(
            vector_class=klass, query_vector=query_vector, limit=max_documents_per_query
        )
        for query_vector in query_vectors
        for klass in query_classes
    ]


def _get_unique_documents(
    *,
    documents: list[list[embedding.EmbeddedChunk]],
) -> list[embedding.EmbeddedChunk]:
    unique_chunks: dict[str, embedding.EmbeddedChunk] = {}

    for document_list in documents:
        for document in document_list:
            if document.id not in unique_chunks.keys():
                unique_chunks[document.id] = document

    return list(unique_chunks.values())
