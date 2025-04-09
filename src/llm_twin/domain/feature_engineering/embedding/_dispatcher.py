import collections
import dataclasses

from llm_twin.domain import models
from llm_twin.domain.feature_engineering import chunking
from llm_twin.domain.storage import vector as vector_storage

from . import _documents, _embedders


@dataclasses.dataclass(frozen=True)
class NoChunkEmbedderRegistered(Exception):
    data_category: vector_storage.DataCategory


ChunkEmbedderT = _embedders.ChunkEmbedder[_embedders.EmbeddedChunkT, _embedders.ChunkT]
EmbedderRegistryT = dict[vector_storage.DataCategory, ChunkEmbedderT]


class EmbedderDispatcher:
    def __init__(self, *, embedding_model: models.EmbeddingModel) -> None:
        self._embedder_registry: EmbedderRegistryT = {
            vector_storage.DataCategory.ARTICLES: _embedders.ArticleChunkEmbedder(
                embedding_model=embedding_model
            ),
            vector_storage.DataCategory.REPOSITORIES: _embedders.RepositoryChunkEmbedder(
                embedding_model=embedding_model
            ),
        }

    def embed_chunk(self, *, chunk: chunking.Chunk) -> _documents.EmbeddedChunk:
        embedded_chunks = self.embed_chunks(chunks=[chunk])
        return embedded_chunks[0]

    def embed_chunks(
        self, *, chunks: list[chunking.Chunk]
    ) -> list[_documents.EmbeddedChunk]:
        grouped_chunks = collections.defaultdict(list)
        for chunk in chunks:
            data_category = chunk.category()
            grouped_chunks[data_category].append(chunk)

        embedded_chunks: list[_documents.EmbeddedChunk] = []

        for data_category, category_chunks in grouped_chunks.items():
            try:
                embedder = self._embedder_registry[data_category]
            except KeyError as exc:
                raise NoChunkEmbedderRegistered(data_category=data_category) from exc

            batch = embedder.embed_batch(chunks=category_chunks)
            embedded_chunks.extend(batch)

        return embedded_chunks
