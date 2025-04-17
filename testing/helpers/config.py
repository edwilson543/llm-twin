import contextlib
import typing
from unittest import mock

from llm_twin import config
from testing.helpers import models as models_helpers
from testing.helpers import storage as storage_helpers


# Databases.


@contextlib.contextmanager
def install_in_memory_document_db(
    db: storage_helpers.InMemoryDocumentDatabase | None = None,
) -> typing.Generator[storage_helpers.InMemoryDocumentDatabase, None, None]:
    """
    Helper to install an in memory database for unit tests.
    """
    db = db or storage_helpers.InMemoryDocumentDatabase()
    with mock.patch.object(config, "get_document_database", return_value=db):
        yield db


@contextlib.contextmanager
def install_in_memory_vector_db(
    db: storage_helpers.InMemoryVectorDatabase | None = None,
) -> typing.Generator[storage_helpers.InMemoryVectorDatabase, None, None]:
    """
    Helper to install an in memory database for unit tests.
    """
    db = db or storage_helpers.InMemoryVectorDatabase()
    with mock.patch.object(config, "get_vector_database", return_value=db):
        yield db


# Models.


@contextlib.contextmanager
def install_fake_embedding_model() -> typing.Generator[
    models_helpers.FakeEmbeddingModel, None, None
]:
    """
    Helper that overrides the settings module to install a fake embedding model.
    """
    fake_embedding_model = models_helpers.get_fake_embedding_model()
    with mock.patch.object(
        config, "get_embedding_model", return_value=fake_embedding_model
    ):
        yield fake_embedding_model
