import typing
from unittest import mock

import pytest

from llm_twin import settings
from testing import settings as testing_settings


@pytest.fixture(scope="session")
def unit_test_settings() -> testing_settings.UnitTestSettings:
    return testing_settings.UnitTestSettings()


@pytest.fixture(scope="session", autouse=True)
def _install_unit_test_settings(
    unit_test_settings,
) -> typing.Generator[None, None, None]:
    with mock.patch.object(settings, "settings", return_value=unit_test_settings):
        yield
