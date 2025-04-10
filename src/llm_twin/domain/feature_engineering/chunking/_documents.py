import abc

import pydantic

from llm_twin.domain.storage import vector as vector_storage


class Chunk(vector_storage.Vector, abc.ABC):
    raw_document_id: str
    content: str
    platform: str
    author_id: str
    author_full_name: str
    metadata: dict = pydantic.Field(default_factory=dict)


class ArticleChunk(Chunk):
    link: str

    class _Config(vector_storage.Config):
        collection = vector_storage.Collection.CLEANED_ARTICLES
        category = vector_storage.DataCategory.ARTICLES


class RepositoryChunk(Chunk):
    name: str
    link: str

    class _Config(vector_storage.Config):
        collection = vector_storage.Collection.CLEANED_REPOSITORIES
        category = vector_storage.DataCategory.REPOSITORIES
