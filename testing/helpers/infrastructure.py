import collections
import contextlib
import dataclasses
import typing
from unittest import mock

from llm_twin import settings
from llm_twin.domain.etl import raw_documents


@dataclasses.dataclass(frozen=True)
class InMemoryRawDocumentDatabase(raw_documents.RawDocumentDatabase):
    _data: dict[raw_documents.Collection, list[raw_documents.SerializedRawDocument]] = (
        dataclasses.field(default_factory=lambda: collections.defaultdict(list))
    )

    @property
    def data(
        self,
    ) -> dict[raw_documents.Collection, list[raw_documents.SerializedRawDocument]]:
        return dict(self._data)

    def find_one(
        self, *, collection: raw_documents.Collection, **filter_options: object
    ) -> raw_documents.SerializedRawDocument:
        collection_documents = self._data.get(collection, [])
        for raw_document in collection_documents:
            if self._document_matches_filter_options(raw_document, **filter_options):
                return raw_document
        raise raw_documents.DocumentDoesNotExist()

    def find_many(
        self, *, collection: raw_documents.Collection, **filter_options: object
    ) -> list[raw_documents.SerializedRawDocument]:
        collection_documents = self._data.get(collection, [])
        return [
            raw_document
            for raw_document in collection_documents
            if self._document_matches_filter_options(raw_document, **filter_options)
        ]

    def insert_one(
        self,
        *,
        collection: raw_documents.Collection,
        document: raw_documents.SerializedRawDocument,
    ) -> None:
        self._data[collection].append(document)

    @staticmethod
    def _document_matches_filter_options(
        document: raw_documents.SerializedRawDocument, **filter_options: object
    ) -> bool:
        return all(
            document.get(filter_key) == filter_value
            for filter_key, filter_value in filter_options.items()
        )


@contextlib.contextmanager
def install_in_memory_raw_document_db(
    db: InMemoryRawDocumentDatabase | None = None,
) -> typing.Generator[InMemoryRawDocumentDatabase, None, None]:
    """
    Helper to install an in memory database for unit tests.
    """
    db = db or InMemoryRawDocumentDatabase()
    with mock.patch.object(settings, "get_raw_document_database", return_value=db):
        yield db
