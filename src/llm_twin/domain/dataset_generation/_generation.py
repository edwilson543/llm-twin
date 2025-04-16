import abc
import dataclasses
from abc import abstractmethod

from llm_twin.domain import models
from llm_twin.domain.storage import vector as vector_storage

from . import _datasets, _prompts


@dataclasses.dataclass(frozen=True)
class SampleDatasetFactory[SampleDatasetT: _datasets.SampleDataset](abc.ABC):
    language_model: models.LanguageModel
    prompt_template_str: str

    @classmethod
    def generate_sample_dataset(
        cls, *, prompts: list[_prompts.GenerateSamplePrompt], test_size: float
    ) -> _datasets.TrainTestSplit[SampleDatasetT]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _post_process_datasets(
        cls,
        datasets: dict[vector_storage.DataCategory, SampleDatasetT],
        test_size: float,
    ) -> _datasets.TrainTestSplit[SampleDatasetT]:
        raise NotImplementedError
