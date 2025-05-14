from llm_twin.domain import inference


class FakeLLMTwin(inference.LLMTwinModelBase):
    def get_response(
        self, *, instruction: str, max_tokens: int, top_k: int | None = None
    ) -> tuple[str, str]:
        return instruction, self.stub_response

    @property
    def stub_response(self) -> str:
        return "response"
