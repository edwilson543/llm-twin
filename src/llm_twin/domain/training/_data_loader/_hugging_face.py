import dataclasses

import datasets

from llm_twin.domain import dataset_generation

from . import _base


@dataclasses.dataclass(frozen=True)
class HuggingFaceDataLoader(_base.DataLoader):
    dataset_path: str
    test_size: float = 0.05
    split: str | None = None

    def load_instruct_dataset(
        self, *, author_id: str
    ) -> dataset_generation.TrainTestSplit[dataset_generation.InstructSample]:
        hf_dataset = datasets.load_dataset(self.dataset_path, split=self.split)

        samples = [
            dataset_generation.InstructSample(
                instruction=sample["instruction"], answer=sample["output"]
            )
            for sample in hf_dataset
        ]

        dataset = dataset_generation.SampleDataset(
            author_id=author_id,
            dataset_type=dataset_generation.DatasetType.INSTRUCT,
            samples=samples,
        )

        return dataset.train_test_split(test_size=self.test_size)

    def load_preference_dataset(
        self, *, author_id: str
    ) -> dataset_generation.TrainTestSplit[dataset_generation.PreferenceSample]:
        hf_dataset = datasets.load_dataset(self.dataset_path, split=self.split)

        samples = [
            dataset_generation.PreferenceSample(
                instruction=sample["prompt"],
                chosen=sample["chosen"],
                rejected=sample["rejected"],
            )
            for sample in hf_dataset
        ]

        dataset = dataset_generation.SampleDataset(
            author_id=author_id,
            dataset_type=dataset_generation.DatasetType.PREFERENCE,
            samples=samples,
        )

        return dataset.train_test_split(test_size=self.test_size)
