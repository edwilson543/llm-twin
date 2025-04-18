import typing

import zenml

from llm_twin.domain import dataset_generation


@zenml.step
def generate_sample_dataset(
    prompts: typing.Annotated[
        list[dataset_generation.GenerateSamplePrompt], "generate_sample_prompts"
    ],
    test_split_size: float,
    dataset_type: typing.Annotated[dataset_generation.DatasetType, "dataset_type"],
) -> typing.Annotated[dataset_generation.TrainTestSplit, "sample_dataset"]:
    prompt_factory = dataset_generation.GenerateSamplePromptFactory()
    system_prompt = prompt_factory.get_system_prompt(dataset_type=dataset_type)
    del system_prompt
    raise NotImplementedError
