import dataclasses

from llm_twin.domain import models


@dataclasses.dataclass(frozen=True)
class FakeLanguageModel(models.LanguageModel):
    def encode(self, *, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, *, tokens: list[int]) -> str:
        return "some text"
