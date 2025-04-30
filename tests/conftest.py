import pathlib
import shutil
import typing
import uuid
from unittest import mock

import pytest

from llm_twin import config
from llm_twin.infrastructure.db import qdrant
from testing.helpers import storage as storage_helpers


@pytest.fixture(scope="function")
def _install_qdrant_db_with_tear_down(monkeypatch) -> None:
    """
    Patch the actual Qdrant database with a subclass that provides tear down functionality.
    """
    monkeypatch.setattr(
        qdrant, "QdrantDatabase", storage_helpers.QdrantDatabaseWithTearDown
    )


@pytest.fixture(scope="function", autouse=True)
def _tear_down_qdrant_db(
    _install_qdrant_db_with_tear_down,
) -> typing.Generator[None, None, None]:
    """
    If the Qdrant database was used, remove all inserted content at the end of the test.

    Note that Qdrant tear down is conditional in case the in memory vector db was used.
    """
    db = config.get_vector_database()

    with mock.patch.object(config, "get_vector_database", return_value=db):
        try:
            yield
        finally:
            if isinstance(db, storage_helpers.QdrantDatabaseWithTearDown):
                db.tear_down()


@pytest.fixture
def output_dir() -> typing.Generator[pathlib.Path, None, None]:
    """
    Create a directory and delete it at the end of the test.
    """
    output_dir = pathlib.Path(__file__).parent / "outputs" / str(uuid.uuid4())
    output_dir.mkdir(parents=True, exist_ok=False)

    try:
        yield output_dir
    finally:
        shutil.rmtree(output_dir)
