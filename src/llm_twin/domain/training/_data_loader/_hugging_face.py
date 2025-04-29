import dataclasses

import datasets

from llm_twin.domain import dataset_generation

from . import _base


@dataclasses.dataclass(frozen=True)
class HuggingFaceDataLoader(_base.DataLoader):
    dataset_path: str
    eos_token: str

    def load(self) -> dataset_generation.TrainTestSplit:
        dataset = datasets.load_dataset(self.dataset_path, split="train")
        return dataset_generation.TrainTestSplit.model_validate(dataset)
