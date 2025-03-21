import pytest

from llm_twin.infrastructure.documents import _mongodb
from testing import settings


@pytest.fixture(scope="session")
def _connector() -> _mongodb.MongoDatabaseConnector:
    return _mongodb.MongoDatabaseConnector(settings=settings.IntegrationTestSettings())


@pytest.fixture(scope="function")
def db(_connector) -> _mongodb.MongoDatabase:
    return _mongodb.MongoDatabase(_connector=_connector)
