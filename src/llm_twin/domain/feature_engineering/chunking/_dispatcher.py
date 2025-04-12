import dataclasses

from llm_twin.domain import models
from llm_twin.domain.storage import vector as vector_storage

from . import _chunkers, _documents


@dataclasses.dataclass(frozen=True)
class NoDocumentChunkerRegistered(Exception):
    data_category: vector_storage.DataCategory


ChunkerT = _chunkers.DocumentChunker[_chunkers.CleanedDocumentT, _chunkers.ChunkT]
ChunkerRegistryT = dict[vector_storage.DataCategory, ChunkerT]


class ChunkerDispatcher:
    def __init__(self, *, embedding_model_config: models.EmbeddingModelConfig) -> None:
        self._chunker_registry: ChunkerRegistryT = {
            vector_storage.DataCategory.ARTICLES: _chunkers.ArticleChunker(
                embedding_model_config=embedding_model_config
            ),
            vector_storage.DataCategory.REPOSITORIES: _chunkers.RepositoryChunker(
                embedding_model_config=embedding_model_config
            ),
        }

    def split_document_into_chunks(
        self, *, document: _chunkers.CleanedDocumentT
    ) -> list[_documents.Chunk]:
        data_category = document.category()

        try:
            chunker = self._chunker_registry[data_category]
        except KeyError as exc:
            raise NoDocumentChunkerRegistered(data_category=data_category) from exc

        return chunker.chunk(document=document)
