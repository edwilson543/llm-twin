import dataclasses

from llm_twin.domain.storage import document as document_storage
from llm_twin.domain.storage import vector as vector_storage

from . import _cleaners, _documents


@dataclasses.dataclass(frozen=True)
class NoCleanerRegistered(Exception):
    collection: document_storage.Collection


CleanerT = _cleaners.DocumentCleaner[_cleaners.RawDocumentT, _cleaners.CleanedDocumentT]
CleanerRegistryT = dict[vector_storage.DataCategory, CleanerT]


class CleanerDispatcher:
    def __init__(self) -> None:
        self._cleaner_registry: CleanerRegistryT = {
            vector_storage.DataCategory.ARTICLES: _cleaners.ArticleCleaner(),
            vector_storage.DataCategory.REPOSITORIES: _cleaners.RepositoryCleaner(),
        }

    def clean_document(
        self, *, document: _cleaners.RawDocumentT
    ) -> _documents.CleanedDocument:
        raw_document_collection = document.get_collection_name()

        try:
            data_category = vector_storage.DataCategory(raw_document_collection.value)
        except ValueError as exc:
            raise NoCleanerRegistered(collection=raw_document_collection) from exc

        try:
            cleaner = self._cleaner_registry[data_category]
        except KeyError as exc:
            raise NoCleanerRegistered(collection=raw_document_collection) from exc

        return cleaner.clean(document=document)
