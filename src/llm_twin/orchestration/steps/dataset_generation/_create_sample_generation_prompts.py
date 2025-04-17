import typing

import zenml

from llm_twin.domain import dataset_generation
from llm_twin.domain.feature_engineering import chunking


@zenml.step
def create_sample_generation_prompts(
    chunks: typing.Annotated[list[chunking.Chunk], "chunked_documents"],
    dataset_type: typing.Annotated[dataset_generation.DatasetType, "dataset_type"],
) -> typing.Annotated[
    list[dataset_generation.GenerateSamplePrompt], "generate_sample_prompts"
]:
    raise NotImplementedError
