import abc
import typing

import pydantic

from llm_twin.domain.storage import vector as vector_storage


class EmbeddedChunk(vector_storage.Vector, abc.ABC):
    content: str
    embedding: list[float]
    platform: str
    chunked_document_id: str
    author_id: str
    author_full_name: str
    metadata: dict = pydantic.Field(default_factory=dict)

    @classmethod
    def to_context(cls, chunks: list[typing.Self]) -> str:
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"""
            Chunk {i + 1}:
            Type: {chunk.__class__.__name__}
            Platform: {chunk.platform}
            Author: {chunk.author_full_name}
            Content: {chunk.content}\n
            """

        return context


class EmbeddedArticleChunk(EmbeddedChunk):
    link: str

    class _Config(vector_storage.Config):
        collection = vector_storage.Collection.EMBEDDED_ARTICLES
        category = vector_storage.DataCategory.ARTICLES


class EmbeddedRepositoryChunk(EmbeddedChunk):
    name: str
    link: str

    class _Config(vector_storage.Config):
        collection = vector_storage.Collection.EMBEDDED_REPOSITORIES
        category = vector_storage.DataCategory.REPOSITORIES
