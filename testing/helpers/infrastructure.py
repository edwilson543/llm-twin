import collections
import contextlib
import dataclasses
import typing
from unittest import mock

from llm_twin import settings
from llm_twin.domain import raw_documents
from llm_twin.domain.raw_documents import RawDocument


@dataclasses.dataclass(frozen=True)
class InMemoryNoSQLDatabase(raw_documents.NoSQLDatabase):
    _data: dict[raw_documents.Collection, list[raw_documents.RawDocument]] = (
        dataclasses.field(default_factory=lambda: collections.defaultdict(list))
    )

    @property
    def data(self) -> dict[raw_documents.Collection, list[raw_documents.RawDocument]]:
        return dict(self._data)

    def find_one(
        self, *, collection: raw_documents.Collection, **filter_options: object
    ) -> raw_documents.RawDocument:
        collection_documents = self._data.get(collection, [])
        for raw_document in collection_documents:
            if all(
                raw_document.get(filter_key) == filter_value
                for filter_key, filter_value in filter_options.items()
            ):
                return raw_document
        raise raw_documents.DocumentDoesNotExist()

    def insert_one(
        self, *, collection: raw_documents.Collection, document: RawDocument
    ) -> None:
        self._data[collection].append(document)


@contextlib.contextmanager
def install_in_memory_db(
    db: InMemoryNoSQLDatabase | None = None,
) -> typing.Generator[InMemoryNoSQLDatabase, None, None]:
    """
    Helper to install an in memory database for unit tests.
    """
    db = db or InMemoryNoSQLDatabase()
    with mock.patch.object(settings, "get_nosql_database", return_value=db):
        yield db
