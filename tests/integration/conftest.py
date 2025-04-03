import typing

import pytest

from llm_twin import settings
from llm_twin.domain.storage import vector as vector_storage
from llm_twin.infrastructure.db import qdrant


@pytest.fixture(scope="function")
def qdrant_db() -> typing.Generator[qdrant.QdrantDatabase, None, None]:
    """
    Provide a qdrant database and remove all testing data after each test.
    """
    db = settings.get_vector_database()
    assert isinstance(db, qdrant.QdrantDatabase)

    try:
        yield db
    finally:
        db.delete_collection(collection=vector_storage.Collection.TESTING_VECTORS)
        db.delete_collection(
            collection=vector_storage.Collection.TESTING_VECTOR_EMBEDDINGS
        )
