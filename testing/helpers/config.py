import contextlib
import typing
from unittest import mock

from llm_twin import config
from llm_twin.config import _dependencies
from llm_twin.domain import rag
from testing.helpers import inference as inference_helpers
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


@contextlib.contextmanager
def install_fake_language_model() -> typing.Generator[
    models_helpers.FakeLanguageModel, None, None
]:
    """
    Helper that overrides the settings module to install a fake language model.
    """
    fake_language_model = models_helpers.FakeLanguageModel()
    with (
        mock.patch.object(
            config, "get_language_model", return_value=fake_language_model
        ),
        mock.patch.object(
            _dependencies, "get_language_model", return_value=fake_language_model
        ),
    ):
        yield fake_language_model


# RAG.


def get_retrieval_config(
    db: storage_helpers.InMemoryVectorDatabase | None = None,
    number_of_query_expansions: int = 1,
    max_documents_per_query: int = 1,
) -> rag.RetrievalConfig:
    return rag.RetrievalConfig(
        db=db or storage_helpers.InMemoryVectorDatabase(),
        language_model=models_helpers.FakeLanguageModel(),
        embedding_model=models_helpers.get_fake_embedding_model(),
        cross_encoder_model=models_helpers.FakeCrossEncoder(),
        number_of_query_expansions=number_of_query_expansions,
        max_documents_per_query=max_documents_per_query,
    )


@contextlib.contextmanager
def install_fake_inference_engine() -> typing.Generator[
    inference_helpers.FakeInferenceEngine, None, None
]:
    inference_engine = inference_helpers.FakeInferenceEngine()
    with mock.patch.object(
        config, "get_inference_engine", return_value=inference_engine
    ):
        yield inference_engine
