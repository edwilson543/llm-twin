import dataclasses

from llm_twin.domain import dataset_generation, models
from testing.factories import dataset as dataset_factories


@dataclasses.dataclass(frozen=True)
class FakeLanguageModel(models.LanguageModel):
    def get_response(
        self,
        *,
        messages: list[models.Message],
        response_format: type[models.ResponseFormatT],
    ) -> models.ResponseFormatT:
        response_factories = {
            dataset_generation.InstructSampleList: dataset_factories.InstructSampleList,
            dataset_generation.PreferenceSampleList: dataset_factories.PreferenceSampleList,
        }

        response = response_factories[response_format]()
        assert isinstance(response, response_format)  # For mypy.

        return response
