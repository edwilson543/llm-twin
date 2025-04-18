import abc
import dataclasses
import enum
import typing

import pydantic


_ResponseFormatT = typing.TypeVar("_ResponseFormatT", bound=pydantic.BaseModel)


class Role(enum.Enum):
    SYSTEM = "system"
    USER = "user"


@dataclasses.dataclass(frozen=True)
class Message:
    content: str
    role: Role

    @classmethod
    def system(cls, *, content: str) -> typing.Self:
        return cls(content=content, role=Role.SYSTEM)

    @classmethod
    def user(cls, *, content: str) -> typing.Self:
        return cls(content=content, role=Role.USER)

    def serialize(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


class LanguageModel(abc.ABC):
    """
    Third-party LLM used to automate tasks in the training pipeline.
    """

    @abc.abstractmethod
    def get_response(
        self, *, messages: list[Message], response_format: type[_ResponseFormatT]
    ) -> _ResponseFormatT:
        raise NotImplementedError

    @abc.abstractmethod
    def encode(self, *, text: str) -> list[int]:
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, *, tokens: list[int]) -> str:
        raise NotImplementedError


class OpenAILanguageModel(LanguageModel):
    def get_response(
        self, *, messages: list[Message], response_format: type[_ResponseFormatT]
    ) -> _ResponseFormatT:
        raise NotImplementedError

    def encode(self, *, text: str) -> list[int]:
        return []

    def decode(self, *, tokens: list[int]) -> str:
        return ""
