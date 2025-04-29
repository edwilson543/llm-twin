import abc
import dataclasses


@dataclasses.dataclass(frozen=True)
class DataLoader(abc.ABC):
    @abc.abstractmethod
    def load(self) -> dict:
        raise NotImplementedError
