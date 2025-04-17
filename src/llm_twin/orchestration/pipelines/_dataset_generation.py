import zenml

from llm_twin.domain import dataset_generation
from llm_twin.orchestration.steps import dataset_generation as dataset_generation_steps


@zenml.pipeline
def generate_sample_datasets(
    *, dataset_type: dataset_generation.DatasetType, test_split_size: float
) -> None:
    chunks = dataset_generation_steps.fetch_chunked_documents()
    prompts = dataset_generation_steps.create_sample_generation_prompts(
        documents=chunks, dataset_type=dataset_type
    )
    dataset_generation_steps.generate_sample_dataset(
        prompts=prompts, test_split_size=test_split_size
    )
