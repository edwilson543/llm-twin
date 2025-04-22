import abc
import dataclasses
import enum
import typing

import openai
import pydantic


ResponseFormatT = typing.TypeVar("ResponseFormatT", bound=pydantic.BaseModel)


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


@dataclasses.dataclass(frozen=True)
class UnableToGetResponse(Exception):
    model: str


class LanguageModel(abc.ABC):
    """
    Third-party LLM used to automate tasks in the training pipeline.
    """

    @abc.abstractmethod
    def get_response(
        self, *, messages: list[Message], response_format: type[ResponseFormatT]
    ) -> ResponseFormatT:
        raise NotImplementedError


class OpenAILanguageModel(LanguageModel):
    def __init__(self, *, api_key: str, model: str) -> None:
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def get_response(
        self, *, messages: list[Message], response_format: type[ResponseFormatT]
    ) -> ResponseFormatT:
        serialized_messages = [message.serialize() for message in messages]

        try:
            response = self._client.beta.chat.completions.parse(
                model=self._model,
                messages=serialized_messages,  # type: ignore[arg-type]
                response_format=response_format,
            )
        except openai.APIError as exc:
            raise UnableToGetResponse(model=self._model) from exc

        if not (parsed_samples := response.choices[0].message.parsed):
            raise UnableToGetResponse(model=self._model)

        return parsed_samples
