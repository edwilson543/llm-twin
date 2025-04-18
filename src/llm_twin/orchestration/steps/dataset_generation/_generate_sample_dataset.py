import typing

import zenml

from llm_twin import config
from llm_twin.domain import dataset_generation
from llm_twin.orchestration.steps import context


@zenml.step
def generate_sample_dataset(
    dataset_type: typing.Annotated[dataset_generation.DatasetType, "dataset_type"],
    prompts: typing.Annotated[
        list[dataset_generation.GenerateSamplePrompt], "generate_sample_prompts"
    ],
    test_size: float,
    context: context.StepContext | None = None,
) -> typing.Annotated[dataset_generation.TrainTestSplit, "sample_dataset"]:
    prompt_factory = dataset_generation.GenerateSamplePromptFactory()
    system_prompt = prompt_factory.get_system_prompt(dataset_type=dataset_type)

    language_model = config.get_language_model()

    dataset = dataset_generation.generate_sample_dataset(
        language_model=language_model,
        system_prompt=system_prompt,
        prompts=prompts,
    )

    step_context = context or zenml.get_step_context()
    step_context.add_output_metadata(
        output_name="sample_dataset",
        metadata={
            "dataset_type": dataset_type.value,
            "num_samples": dataset.num_samples,
        },
    )

    return dataset.train_test_split(test_size=test_size)
