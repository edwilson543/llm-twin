import zenml

from llm_twin.domain import dataset_generation
from llm_twin.orchestration.steps import dataset_generation as dataset_generation_steps


@zenml.pipeline
def generate_sample_dataset(
    *,
    author_id: str,
    dataset_type: dataset_generation.DatasetType,
    test_size: float,
) -> None:
    chunks = dataset_generation_steps.fetch_chunked_documents(author_id=author_id)
    prompts = dataset_generation_steps.create_prompts_for_generating_samples(
        documents=chunks, dataset_type=dataset_type
    )
    dataset_generation_steps.generate_sample_dataset(
        author_id=author_id,
        dataset_type=dataset_type,
        prompts=prompts,
        test_size=test_size,
    )
