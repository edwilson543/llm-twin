import typing

import zenml

from llm_twin.domain import dataset_generation
from llm_twin.domain.feature_engineering import chunking
from llm_twin.orchestration.steps import context


@zenml.step
def create_prompts_for_generating_samples(
    documents: typing.Annotated[list[chunking.Chunk], "chunked_documents"],
    dataset_type: typing.Annotated[dataset_generation.DatasetType, "dataset_type"],
    context: context.StepContext | None = None,
) -> typing.Annotated[
    list[dataset_generation.GenerateSamplePrompt], "generate_sample_prompts"
]:
    prompt_factory = dataset_generation.GenerateSamplePromptFactory()
    prompts = prompt_factory.create_prompts_for_generating_samples(
        dataset_type=dataset_type, documents=documents
    )

    step_context = context or zenml.get_step_context()
    step_context.add_output_metadata(
        output_name="sample_generation_prompts", metadata={"num_prompts": len(prompts)}
    )

    return prompts
