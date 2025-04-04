from abc import ABC

from llm_twin.domain.storage import vector as vector_storage


class CleanedDocument(vector_storage.Vector, ABC):
    content: str
    platform: str
    author_id: str
    author_full_name: str


class CleanedArticle(CleanedDocument):
    link: str

    class _Config(vector_storage.Config):
        collection = vector_storage.Collection.CLEANED_ARTICLES
        category = vector_storage.DataCategory.ARTICLES


class CleanedRepository(CleanedDocument):
    name: str
    link: str

    class _Config(vector_storage.Config):
        collection = vector_storage.Collection.CLEANED_REPOSITORIES
        category = vector_storage.DataCategory.REPOSITORIES
