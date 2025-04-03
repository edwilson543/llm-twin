import dataclasses

from llm_twin.domain.storage import vector as vector_storage


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

    def bulk_insert(self, *, vectors: list[vector_storage.Vector]) -> None:
        self.vectors.extend(vectors)
