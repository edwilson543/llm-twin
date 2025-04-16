import dataclasses

from llm_twin.domain import models
from llm_twin.domain.feature_engineering import chunking
from llm_twin.domain.storage import vector as vector_storage


class Prompt(vector_storage.Vector):
    template: str
    input_variables: dict
    content: str
    num_tokens: int | None = None

    class _Config(vector_storage.Config):
        category = vector_storage.DataCategory.PROMPT


class GenerateSamplePrompt(vector_storage.Vector):
    input_data_category: vector_storage.DataCategory
    document: chunking.Chunk


@dataclasses.dataclass(frozen=True)
class GenerateSamplePromptFactory:
    language_model: models.LanguageModel
    prompt_template_str: str

    @classmethod
    def get_generate_sample_prompts(
        cls, *, documents: list[chunking.Chunk]
    ) -> list[GenerateSamplePrompt]:
        raise NotImplementedError

    @classmethod
    def _get_system_prompt(cls) -> Prompt:
        template = "You are a helpful assistant who generates {dataset_format} based on the given context."
        del template
        raise NotImplementedError

    @classmethod
    def _get_prompt(cls, document: chunking.Chunk) -> GenerateSamplePrompt:
        raise NotImplementedError
