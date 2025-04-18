import dataclasses

from llm_twin.domain import models
from llm_twin.domain.models import Message
from llm_twin.domain.models._language import _ResponseFormatT


@dataclasses.dataclass(frozen=True)
class FakeLanguageModel(models.LanguageModel):
    def get_response(
        self, *, messages: list[Message], response_format: type[_ResponseFormatT]
    ) -> _ResponseFormatT:
        return response_format()

    def encode(self, *, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, *, tokens: list[int]) -> str:
        return "some text"
