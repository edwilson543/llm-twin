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
