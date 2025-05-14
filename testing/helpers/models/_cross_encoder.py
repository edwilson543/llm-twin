from llm_twin.domain import models


class FakeCrossEncoder(models.CrossEncoderModel):
    def predict(self, *, pairs: list[tuple[str, str]]) -> list[float]:
        return [1.23 for _ in pairs]
