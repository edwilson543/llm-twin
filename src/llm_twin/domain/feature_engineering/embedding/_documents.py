import abc

import pydantic

from llm_twin.domain.storage import vector as vector_storage


class EmbeddedChunk(vector_storage.VectorEmbedding, abc.ABC):
    raw_document_id: str
    content: str
    platform: str
    author_id: str
    author_full_name: str
    metadata: dict = pydantic.Field(default_factory=dict)

    def to_context(self) -> str:
        return f"""Category: {self.category().value}
            Platform: {self.platform}
            Author: {self.author_full_name}
            Content: {self.content}"""


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
