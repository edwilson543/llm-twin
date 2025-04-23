from __future__ import annotations

import enum

from sklearn.model_selection import train_test_split

from llm_twin.domain.storage import vector as vector_storage


class DatasetType(enum.StrEnum):
    INSTRUCT = "INSTRUCT"
    PREFERENCE = "PREFERENCE"


# Samples.


class InstructSample(vector_storage.Vector):
    instruction: str
    answer: str

    class _Config(vector_storage.Config):
        category = vector_storage.DataCategory.INSTRUCT_SAMPLE


class PreferenceSample(vector_storage.Vector):
    instruction: str
    rejected: str
    chosen: str

    class _Config(vector_storage.Config):
        category = vector_storage.DataCategory.PREFERENCE_SAMPLE


SampleT = InstructSample | PreferenceSample

# Datasets.


class SampleDataset(vector_storage.Vector):
    samples: list[SampleT]

    def train_test_split(
        self,
        test_size: float,
        random_state: int = 42,
    ) -> TrainTestSplit:
        train_samples, test_samples = train_test_split(
            self.samples, test_size=test_size, random_state=random_state
        )
        return TrainTestSplit(
            train=SampleDataset(samples=train_samples),
            test=SampleDataset(samples=test_samples),
        )

    @property
    def num_samples(self) -> int:
        return len(self.samples)


class TrainTestSplit(vector_storage.Vector):
    train: SampleDataset
    test: SampleDataset

    @property
    def test_size(self) -> float:
        return self.test.num_samples / (self.train.num_samples + self.test.num_samples)

    @property
    def all_samples(self) -> list[SampleT]:
        return self.train.samples + self.test.samples
