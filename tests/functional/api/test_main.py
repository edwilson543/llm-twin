import pytest
from fastapi import testclient

from llm_twin.interfaces.api import main
from testing.helpers import config as config_helpers


@pytest.fixture
def api_client() -> testclient.TestClient:
    return testclient.TestClient(app=main.app)


class TestCompletions:
    def test_returns_completion_for_simple_query(self, api_client):
        request = {"query": "Are you an LLM?", "max_tokens": 250}

        with (
            config_helpers.install_fake_inference_engine() as engine,
            config_helpers.install_fake_language_model(),
        ):
            response = api_client.post("/completions", json=request)

        assert response.status_code == 200
        assert response.json() == {"completion": engine.stub_response}

    def test_returns_bad_response_when_request_is_malformed(self, api_client):
        request = {"question": "Are you an LLM?"}

        response = api_client.post("/completions", json=request)

        assert response.status_code == 422
