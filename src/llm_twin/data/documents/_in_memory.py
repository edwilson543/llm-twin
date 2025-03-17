import collections
import dataclasses

from llm_twin.domain import documents
from llm_twin.domain.documents import RawDocument


@dataclasses.dataclass(frozen=True)
class InMemoryNoSQLDatabase(documents.NoSQLDatabase):
    _data: dict[str, list[documents.RawDocument]] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(list)
    )

    def find_one(
        self, *, collection: str, **filter_options: object
    ) -> documents.RawDocument:
        collection_documents = self._data.get(collection, [])
        for raw_document in collection_documents:
            if all(
                raw_document.get(filter_key) == filter_value
                for filter_key, filter_value in filter_options.items()
            ):
                return raw_document
        raise documents.DocumentDoesNotExist()

    def insert_one(self, *, collection: str, document: RawDocument) -> None:
        self._data[collection].append(document)
