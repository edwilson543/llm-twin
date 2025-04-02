from __future__ import annotations

import abc
import typing

from . import _vector


VectorT = typing.TypeVar("VectorT", bound=_vector.Vector)
_SerializedVectorT = typing.TypeVar("_SerializedVectorT")


class VectorDatabase[SerializedVectorT](abc.ABC):
    # Read operations.

    @abc.abstractmethod
    def bulk_find(
        self, *, collection: _vector.Collection, limit: int
    ) -> list[_vector.Vector]:
        """
        Find all vectors in the collection matching the filter options.
        """
        raise NotImplementedError

    # Write operations.

    @abc.abstractmethod
    def bulk_insert(self, *, vectors: list[_vector.Vector]) -> None:
        """
        Bulk insert some vectors into the database.
        """
        raise NotImplementedError

    # Serialization.

    @classmethod
    @abc.abstractmethod
    def _deserialize(
        cls, serialized_vector: SerializedVectorT, vector_class: type[VectorT]
    ) -> VectorT:
        raise NotImplementedError

    @abc.abstractmethod
    def _serialize(self, vector: _vector.Vector) -> SerializedVectorT:
        raise NotImplementedError
