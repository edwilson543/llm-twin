import dataclasses

import datasets

from llm_twin.domain import dataset_generation

from . import _base


@dataclasses.dataclass(frozen=True)
class HuggingFaceDataLoader(_base.DataLoader):
    dataset_path: str

    def load_instruct_dataset(
        self,
    ) -> dataset_generation.TrainTestSplit[dataset_generation.InstructSample]:
        dataset = datasets.load_dataset(self.dataset_path, split="train")
        return dataset_generation.TrainTestSplit.model_validate(dataset)
