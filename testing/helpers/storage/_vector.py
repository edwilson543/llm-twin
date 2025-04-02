import dataclasses

from llm_twin.domain.storage import vector as vector_storage
from llm_twin.domain.storage.vector import _vector


@dataclasses.dataclass(frozen=True)
class InMemoryVectorDatabase(vector_storage.VectorDatabase[vector_storage.Vector]):
    vectors: list[vector_storage.Vector] = dataclasses.field(default_factory=list)

    def bulk_find(
        self, *, collection: _vector.Collection, limit: int
    ) -> list[_vector.Vector]:
        vectors = [vector for vector in self.vectors if vector.collection == collection]
        return vectors[:limit]

    def bulk_insert(self, *, vectors: list[_vector.Vector]) -> None:
        self.vectors.extend(vectors)

    @classmethod
    def _deserialize(
        cls,
        serialized_vector: vector_storage.Vector,
        vector_class: type[vector_storage.VectorT],
    ) -> vector_storage.VectorT:
        assert isinstance(serialized_vector, vector_class)  # For mypy.
        return serialized_vector

    def _serialize(self, vector: vector_storage.Vector) -> vector_storage.Vector:
        return vector
