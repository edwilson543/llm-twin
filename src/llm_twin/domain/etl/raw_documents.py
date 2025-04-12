import abc

from llm_twin.domain.storage import document as document_storage


class RawDocument(document_storage.Document, abc.ABC):
    """
    A raw document that was extracted from some webpage by a crawler.
    """

    content: dict
    platform: str
    author_id: str
    author_full_name: str


class Article(RawDocument):
    link: str

    @classmethod
    def get_collection_name(cls) -> document_storage.Collection:
        return document_storage.Collection.ARTICLES


class Repository(RawDocument):
    name: str
    link: str

    @classmethod
    def get_collection_name(cls) -> document_storage.Collection:
        return document_storage.Collection.REPOSITORIES
