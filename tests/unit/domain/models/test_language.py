import typing

import pytest

from llm_twin.domain import dataset_generation
from llm_twin.domain.models import _language as language_models
from testing.factories import dataset as dataset_factories


class TestOpenAILanguageModel__GetResponse:
    def test_gets_structured_response_when_vendor_response_okay(self, httpx_mock):
        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
            status_code=200,
            json=self._sample_response_json(),
        )

        model = language_models.OpenAILanguageModel(api_key="", model="")
        messages = [
            language_models.Message.system(content="hello"),
            language_models.Message.user(content="world"),
        ]

        sample_list = model.get_response(
            messages=messages, response_format=dataset_generation.InstructSampleList
        )

        assert isinstance(sample_list, dataset_generation.InstructSampleList)
        assert len(sample_list.samples) == 5

    def test_raises_error_when_vendor_response_bad(self, httpx_mock):
        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
            status_code=401,
        )
        model = language_models.OpenAILanguageModel(api_key="", model="some-model")

        with pytest.raises(language_models.UnableToGetResponse) as exc:
            model.get_response(
                messages=[], response_format=dataset_generation.InstructSampleList
            )

        assert exc.value.model == "some-model"

    @staticmethod
    def _sample_response_json() -> dict[str, typing.Any]:
        """
        OpenAI chat completion structure response output, per the docs.
        https://platform.openai.com/docs/guides/structured-outputs
        """
        sample_list = dataset_factories.InstructSampleList.build()

        return {
            "id": "chatcmpl-AyCk92plBjBPBOnrcpYQ4xVC7LPZd",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": sample_list.model_dump_json(),
                        "role": "assistant",
                        "parsed": sample_list.model_dump(),
                        "refusal": None,
                    },
                }
            ],
        }
