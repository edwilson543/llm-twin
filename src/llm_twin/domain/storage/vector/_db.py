from __future__ import annotations

import abc
import dataclasses
import typing

from . import _vector


VectorT = typing.TypeVar("VectorT", bound=_vector.Vector)


@dataclasses.dataclass
class UnableToInsertVectors(Exception):
    collection: _vector.Collection


class VectorDatabase(abc.ABC):
    # Read operations.

    @abc.abstractmethod
    def bulk_find(
        self, *, vector_class: type[VectorT], limit: int, offset: str | None = None
    ) -> tuple[list[VectorT], str | None]:
        """
        Find all vectors in the collection matching the filter options.

        :return:
            - The list of retrieved vectors
            - The offset to scroll from to find the next batch of vectors.
        """
        raise NotImplementedError

    # Write operations.

    @abc.abstractmethod
    def bulk_insert(self, *, vectors: typing.Sequence[_vector.Vector]) -> None:
        """
        Bulk insert some vectors into the database.
        """
        raise NotImplementedError
