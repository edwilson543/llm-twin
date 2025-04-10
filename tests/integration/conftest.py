import dataclasses
import typing
from unittest import mock

import pytest

from llm_twin import settings
from llm_twin.domain.storage import vector as vector_storage
from llm_twin.infrastructure.db import qdrant


@dataclasses.dataclass(frozen=True)
class QdrantDatabase(qdrant.QdrantDatabase):
    """
    Thin wrapper around the actual Qdrant database to provide teardown.
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


@pytest.fixture(scope="function", autouse=True)
def qdrant_db() -> typing.Generator[qdrant.QdrantDatabase, None, None]:
    """
    Provide a qdrant database and remove all testing data after each test.
    """
    embedding_model_config = embeddings_helpers.get_fake_embedding_model_config()
    db = QdrantDatabase.build(
        host=settings.settings.QDRANT_DATABASE_HOST,
        port=settings.settings.QDRANT_DATABASE_PORT,
        embedding_model_config=embedding_model_config,
    )

    with mock.patch.object(settings, "get_vector_database", return_value=db):
        try:
            yield db
        finally:
            db.tear_down()
