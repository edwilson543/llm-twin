import abc
import dataclasses

from llm_twin.domain import dataset_generation


@dataclasses.dataclass(frozen=True)
class DataLoader(abc.ABC):
    @abc.abstractmethod
    def load(self) -> dataset_generation.TrainTestSplit:
        raise NotImplementedError
