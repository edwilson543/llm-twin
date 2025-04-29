import abc
import dataclasses

from llm_twin.domain import dataset_generation


@dataclasses.dataclass(frozen=True)
class DataLoader(abc.ABC):
    @abc.abstractmethod
    def load_instruct_dataset(
        self,
    ) -> dataset_generation.TrainTestSplit[dataset_generation.InstructSample]:
        raise NotImplementedError

    @abc.abstractmethod
    def load_preference_dataset(
        self,
    ) -> dataset_generation.TrainTestSplit[dataset_generation.PreferenceSample]:
        raise NotImplementedError
