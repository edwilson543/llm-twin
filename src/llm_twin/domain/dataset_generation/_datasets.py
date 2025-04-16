import typing

from sklearn.model_selection import train_test_split

from llm_twin.domain.storage import vector as vector_storage


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


# Datasets.

SampleT = typing.TypeVar("SampleT", bound=InstructSample | PreferenceSample)


class _SampleDataset(vector_storage.Vector, typing.Generic[SampleT]):
    input_data_category: vector_storage.DataCategory
    samples: list[SampleT]

    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[typing.Self, typing.Self]:
        train_samples, test_samples = train_test_split(
            self.samples, test_size=test_size, random_state=random_state
        )
        return self._new(samples=train_samples), self._new(samples=test_samples)

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def _new(self, *, samples: list[SampleT]) -> typing.Self:
        return type(self)(input_data_category=self.input_data_category, samples=samples)


class InstructSampleDataset(_SampleDataset[InstructSample]):
    class _Config(vector_storage.Config):
        category = vector_storage.DataCategory.INSTRUCT_DATASET


class PreferenceSampleDataset(_SampleDataset[PreferenceSample]):
    class _Config(vector_storage.Config):
        category = vector_storage.DataCategory.PREFERENCE_DATASET


# Train test splits.


class _TrainTestSplit[SampleDatasetT: _SampleDataset](vector_storage.Vector):
    datasets: list[SampleDatasetT]


class InstructTrainTestSplit(_TrainTestSplit[InstructSampleDataset]):
    pass


class PreferenceTrainTestSplit(_TrainTestSplit[PreferenceSampleDataset]):
    pass
