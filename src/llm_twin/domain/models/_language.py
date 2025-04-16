import abc


class LanguageModel(abc.ABC):
    """
    Third-party LLM used to automate tasks in the training pipeline.
    """

    # TODO.
    # @abc.abstractmethod
    # def batch(self, *, text: str, output: type[T]) -> list[T]:
    #     raise NotImplementedError

    @abc.abstractmethod
    def encode(self, *, text: str) -> list[int]:
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, *, tokens: list[int]) -> str:
        raise NotImplementedError
