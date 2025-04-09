from __future__ import annotations

import collections
import dataclasses
import typing

import loguru
import qdrant_client
from qdrant_client import models as qdrant_models
from qdrant_client.http import exceptions as qdrant_exceptions

from llm_twin.domain.storage import vector as vector_storage
from llm_twin.domain.storage.vector import VectorT, _vector


class QdrantDatabaseConnector:
    _client: qdrant_client.QdrantClient

    def __new__(
        cls, database_host: str, database_port: int, *args: object, **kwargs: object
    ) -> QdrantDatabaseConnector:
        if not hasattr(cls, "_client"):
            uri = f"{database_host}:{database_port}"
            cls._client = qdrant_client.QdrantClient(
                host=database_host, port=database_port
            )
            loguru.logger.info(f"Connected to Qdrant DB at: {uri}")

        return super().__new__(cls)

    @property
    def client(self) -> qdrant_client.QdrantClient:
        return self._client


@dataclasses.dataclass(frozen=True)
class QdrantDatabase(vector_storage.VectorDatabase):
    _connector: QdrantDatabaseConnector

    def bulk_find(
        self,
        *,
        vector_class: type[vector_storage.VectorT],
        limit: int,
        offset: str | None = None,
    ) -> tuple[list[vector_storage.VectorT], str | None]:
        collection = vector_class.collection()

        try:
            records, next_offset = self._connector.client.scroll(
                collection_name=collection.value,
                limit=limit,
                with_payload=True,
                with_vectors=True,
                offset=offset,
            )
        except qdrant_exceptions.UnexpectedResponse:
            loguru.logger.exception(
                f"Failed to search documents in '{collection.value}'."
            )
            raise

        documents = [
            _get_vector_from_record(record=record, vector_class=vector_class)
            for record in records
        ]
        next_offset = str(next_offset) if next_offset else None
        return documents, next_offset

    def bulk_insert(self, *, vectors: typing.Sequence[vector_storage.Vector]) -> None:
        grouped_vectors: dict[
            type[vector_storage.Vector], list[qdrant_models.PointStruct]
        ] = collections.defaultdict(list)
        for vector in vectors:
            point = _get_point_from_vector(vector)
            grouped_vectors[type(vector)].append(point)

        for vector_class, points in grouped_vectors.items():
            collection = self._maybe_create_collection(vector_class=vector_class)

            try:
                self._connector.client.upsert(
                    collection_name=collection.value, points=points
                )
            except qdrant_exceptions.UnexpectedResponse as exc:
                raise vector_storage.UnableToInsertVectors(
                    collection=collection
                ) from exc

    def _maybe_create_collection(
        self, vector_class: type[vector_storage.Vector]
    ) -> vector_storage.Collection:
        collection = vector_class.collection()

        if self._connector.client.collection_exists(collection_name=collection.value):
            return collection

        vector_config: dict | qdrant_models.VectorParams = {}
        if vector_class.model_fields.get("embedding"):
            vector_config = qdrant_models.VectorParams(
                size=3, distance=qdrant_models.Distance.COSINE
            )

        self._connector.client.create_collection(
            collection_name=collection.value, vectors_config=vector_config
        )

        return collection


# Serialization.


def _get_vector_from_record(
    *, record: qdrant_models.Record, vector_class: type[VectorT]
) -> VectorT:
    attributes = {"id": record.id, **(record.payload or {})}
    if issubclass(vector_class, vector_storage.VectorEmbedding):
        attributes.update({"embedding": record.vector})
    return vector_class(**attributes)


def _get_point_from_vector(vector: _vector.Vector) -> qdrant_models.PointStruct:
    payload = vector.model_dump(
        exclude={"id", "embedding"}, exclude_unset=False, by_alias=True
    )

    point_vector: list[float] | dict = {}
    if isinstance(vector, vector_storage.VectorEmbedding):
        point_vector = (
            vector.embedding
            if isinstance(vector, vector_storage.VectorEmbedding)
            else {}
        )

    return qdrant_models.PointStruct(id=vector.id, vector=point_vector, payload=payload)
