import typing

import zenml

from llm_twin import config
from llm_twin.domain import dataset_generation
from llm_twin.orchestration.steps import context
from llm_twin.orchestration.steps import types as step_types


@zenml.step
def create_prompts_for_generating_samples(
    documents: step_types.ChunkedDocumentsInputT,
    dataset_type: typing.Annotated[dataset_generation.DatasetType, "dataset_type"],
    context: context.StepContext | None = None,
) -> typing.Annotated[
    list[dataset_generation.GenerateSamplePrompt], "generate_sample_prompts"
]:
    prompt_factory = dataset_generation.GenerateSamplePromptFactory()
    prompts = prompt_factory.create_prompts_for_generating_samples(
        dataset_type=dataset_type, documents=documents
    )

    db = config.get_vector_database()
    db.bulk_insert(vectors=prompts)

    step_context = context or zenml.get_step_context()
    step_context.add_output_metadata(
        output_name="generate_sample_prompts", metadata={"num_prompts": len(prompts)}
    )

    return prompts
