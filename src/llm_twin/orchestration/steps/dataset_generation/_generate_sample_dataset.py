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
    raise NotImplementedError
