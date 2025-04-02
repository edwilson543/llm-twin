from __future__ import annotations

from llm_twin.domain.storage import document as document_storage


class Author(document_storage.RawDocument):
    first_name: str
    last_name: str

    @classmethod
    def get_collection_name(cls) -> document_storage.Collection:
        return document_storage.Collection.AUTHORS

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
