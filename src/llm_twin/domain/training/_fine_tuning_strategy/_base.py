import abc
import dataclasses

from llm_twin.domain.training import _data_loader


@dataclasses.dataclass
class FineTuningStrategy[_TrainingParametersT](abc.ABC):
    data_loader: _data_loader.DataLoader

    @abc.abstractmethod
    def fine_tune(self) -> None:
        raise NotImplementedError
