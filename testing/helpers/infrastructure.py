import collections
import contextlib
import dataclasses
import typing
from unittest import mock

from llm_twin import settings
from llm_twin.domain.storage import document as document_storage


@dataclasses.dataclass(frozen=True)
class InMemoryDocumentDatabase(document_storage.DocumentDatabase):
    _data: dict[
        document_storage.Collection, list[document_storage.SerializedDocument]
    ] = dataclasses.field(default_factory=lambda: collections.defaultdict(list))

    @property
    def data(
        self,
    ) -> dict[document_storage.Collection, list[document_storage.SerializedDocument]]:
        return dict(self._data)

    def find_one(
        self, *, collection: document_storage.Collection, **filter_options: object
    ) -> document_storage.SerializedDocument:
        collection_documents = self._data.get(collection, [])
        for raw_document in collection_documents:
            if self._document_matches_filter_options(raw_document, **filter_options):
                return raw_document
        raise document_storage.DocumentDoesNotExist()

    def find_many(
        self, *, collection: document_storage.Collection, **filter_options: object
    ) -> list[document_storage.SerializedDocument]:
        collection_documents = self._data.get(collection, [])
        return [
            raw_document
            for raw_document in collection_documents
            if self._document_matches_filter_options(raw_document, **filter_options)
        ]

    def insert_one(
        self,
        *,
        collection: document_storage.Collection,
        document: document_storage.SerializedDocument,
    ) -> None:
        self._data[collection].append(document)

    @staticmethod
    def _document_matches_filter_options(
        document: document_storage.SerializedDocument, **filter_options: object
    ) -> bool:
        return all(
            document.get(filter_key) == filter_value
            for filter_key, filter_value in filter_options.items()
        )


@contextlib.contextmanager
def install_in_memory_raw_document_db(
    db: InMemoryDocumentDatabase | None = None,
) -> typing.Generator[InMemoryDocumentDatabase, None, None]:
    """
    Helper to install an in memory database for unit tests.
    """
    db = db or InMemoryDocumentDatabase()
    with mock.patch.object(settings, "get_raw_document_database", return_value=db):
        yield db
