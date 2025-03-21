import typing
from unittest import mock

import pytest

from llm_twin import settings
from testing import settings as testing_settings


@pytest.fixture(scope="session", autouse=True)
def _install_functional_test_settings() -> typing.Generator[None, None, None]:
    functional_test_settings = testing_settings.FunctionalTestSettings()
    with mock.patch.object(settings, "settings", return_value=functional_test_settings):
        yield
