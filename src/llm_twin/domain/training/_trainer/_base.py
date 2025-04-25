import abc
import dataclasses
import enum


class UnableToRunFineTuning(Exception):
    pass


class FineTuneType(enum.Enum):
    SUPERVISED_FINE_TUNING = "SUPERVISED_FINE_TUNING"
    DIRECT_PREFERENCE_OPTIMISATION = "DIRECT_PREFERENCE_OPTIMISATION"


@dataclasses.dataclass(frozen=True)
class Trainer(abc.ABC):
    @abc.abstractmethod
    def run_fine_tuning(
        self,
        *,
        fine_tune_type: FineTuneType,
        num_train_epochs: int,
        per_device_train_batch_size: int,
        learning_rate: float,
    ) -> None:
        raise NotImplementedError
