import abc
import typing

from llm_twin.domain.feature_engineering import chunking

from . import _documents, _model


ChunkT = typing.TypeVar("ChunkT", bound=chunking.Chunk)
EmbeddedChunkT = typing.TypeVar("EmbeddedChunkT", bound=_documents.EmbeddedChunk)


class ChunkEmbedder(abc.ABC, typing.Generic[ChunkT, EmbeddedChunkT]):
    """
    Base class for embedding a particular chunk type.
    """

    def __init__(self, *, embedding_model: _model.EmbeddingModel) -> None:
        self._embedding_model = embedding_model

    def embed(self, *, chunk: ChunkT) -> EmbeddedChunkT:
        embeddings = self.embed_batch(chunks=[chunk])
        return embeddings[0]

    def embed_batch(self, *, chunks: list[ChunkT]) -> list[EmbeddedChunkT]:
        input_text = [chunk.content for chunk in chunks]
        embeddings = self._embedding_model.generate_embeddings(input_text=input_text)

        return [
            self._create_model(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

    @abc.abstractmethod
    def _create_model(self, *, chunk: ChunkT, embedding: list[float]) -> EmbeddedChunkT:
        raise NotImplementedError

    @property
    def _metadata(self) -> dict[str, typing.Any]:
        return {
            "embedding_model_id": self._embedding_model.model_name.value,
            "embedding_size": self._embedding_model.embedding_size,
            "max_input_length": self._embedding_model.max_input_length,
        }


class ArticleChunkEmbedder(
    ChunkEmbedder[chunking.ArticleChunk, _documents.EmbeddedArticleChunk]
):
    def _create_model(
        self, *, chunk: chunking.ArticleChunk, embedding: list[float]
    ) -> _documents.EmbeddedArticleChunk:
        return _documents.EmbeddedArticleChunk(
            id=chunk.id,
            content=chunk.content,
            embedding=embedding,
            platform=chunk.platform,
            link=chunk.link,
            chunked_document_id=chunk.id,
            author_id=chunk.author_id,
            author_full_name=chunk.author_full_name,
            metadata=self._metadata,
        )


class RepositoryChunkEmbedder(
    ChunkEmbedder[chunking.RepositoryChunk, _documents.EmbeddedRepositoryChunk]
):
    def _create_model(
        self, *, chunk: chunking.RepositoryChunk, embedding: list[float]
    ) -> _documents.EmbeddedRepositoryChunk:
        return _documents.EmbeddedRepositoryChunk(
            id=chunk.id,
            content=chunk.content,
            embedding=embedding,
            platform=chunk.platform,
            name=chunk.name,
            link=chunk.link,
            chunked_document_id=chunk.id,
            author_id=chunk.author_id,
            author_full_name=chunk.author_full_name,
            metadata=self._metadata,
        )
