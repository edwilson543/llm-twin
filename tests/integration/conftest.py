import typing
from unittest import mock

import pytest

from llm_twin import settings
from llm_twin.infrastructure.db import qdrant
from testing.helpers import storage as storage_helpers


@pytest.fixture(scope="function", autouse=True)
def qdrant_db() -> typing.Generator[qdrant.QdrantDatabase, None, None]:
    """
    Provide a qdrant database and remove all testing data after each test.
    """
    db = storage_helpers.QdrantDatabase.build(
        host=settings.settings.QDRANT_DATABASE_HOST,
        port=settings.settings.QDRANT_DATABASE_PORT,
        embedding_model_config=settings.get_embedding_model_config(),
    )

    with mock.patch.object(settings, "get_vector_database", return_value=db):
        try:
            yield db
        finally:
            db.tear_down()
