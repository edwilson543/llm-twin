from llm_twin.domain.storage import document as document_storage

from . import _base


class Repository(_base.ExtractedDocument):
    name: str
    link: str

    @classmethod
    def get_collection_name(cls) -> document_storage.Collection:
        return document_storage.Collection.REPOSITORIES
