from unittest import mock

import pytest

from llm_twin.data.documents import _config, _in_memory, _mongodb
from testing import settings


class TestGetNoSQLDatabase:
    def test_gets_mongodb_backend_for_integration_tests(self):
        # Note this doesn't actually connect to the database, hence is still an integration test.
        database = _config.get_nosql_database(
            settings=settings.IntegrationTestSettings()
        )

        assert isinstance(database, _mongodb.MongoDatabase)

    def test_gets_in_memory_backend_for_unit_tests(self):
        database = _config.get_nosql_database(settings=settings.UnitTestSettings())

        assert isinstance(database, _in_memory.InMemoryNoSQLDatabase)

    def test_raises_when_backend_not_configured(self):
        settings = mock.Mock(NOSQL_BACKEND="NOT_CONFIGURED")

        with pytest.raises(_config._NoSQLDatabaseConfigurationError) as exc:
            _config.get_nosql_database(settings=settings)

        assert exc.value.backend == "NOT_CONFIGURED"
