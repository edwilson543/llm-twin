import dataclasses
import typing

from llm_twin.domain.storage import vector as vector_storage
from llm_twin.infrastructure.db import qdrant


@dataclasses.dataclass(frozen=True)
class InMemoryVectorDatabase(vector_storage.VectorDatabase):
    vectors: list[vector_storage.Vector] = dataclasses.field(default_factory=list)

    def bulk_find(
        self,
        *,
        vector_class: type[vector_storage.VectorT],
        limit: int,
        offset: str | None = None,
    ) -> tuple[list[vector_storage.VectorT], str | None]:
        vectors = [
            vector for vector in self.vectors if isinstance(vector, vector_class)
        ]
        return vectors[:limit], None

    def bulk_insert(self, *, vectors: typing.Sequence[vector_storage.Vector]) -> None:
        self.vectors.extend(vectors)

    @property
    def vectors_by_id(self) -> dict[str, vector_storage.Vector]:
        return {vector.id: vector for vector in self.vectors}

    @property
    def vector_ids(self) -> list[str]:
        return list(self.vectors_by_id.keys())


@dataclasses.dataclass(frozen=True)
class QdrantDatabaseWithTearDown(qdrant.QdrantDatabase):
    """
    Thin wrapper around the actual Qdrant database to provide teardown functionality.
    """

    _collections: list[vector_storage.Collection] = dataclasses.field(
        default_factory=list, init=False
    )

    def _maybe_create_collection(
        self, vector_class: type[vector_storage.Vector]
    ) -> vector_storage.Collection:
        """
        Track the collections created during the test, in memory.
        """
        collection = super()._maybe_create_collection(vector_class=vector_class)
        self._collections.append(collection)
        return collection

    def tear_down(self) -> None:
        """
        Delete any collections that were created during the test
        """
        for collection in self._collections:
            self._connector.client.delete_collection(collection_name=collection.value)
