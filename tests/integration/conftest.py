import typing
from unittest import mock

import pytest

from llm_twin import settings
from testing import settings as testing_settings


@pytest.fixture(scope="session")
def integration_test_settings() -> testing_settings.IntegrationTestSettings:
    return testing_settings.IntegrationTestSettings()


@pytest.fixture(scope="session", autouse=True)
def _install_integration_test_settings(
    integration_test_settings,
) -> typing.Generator[None, None, None]:
    with mock.patch.object(
        settings, "settings", return_value=integration_test_settings
    ):
        yield
