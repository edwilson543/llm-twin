import pytest

from llm_twin.domain import dataset_generation
from llm_twin.orchestration.steps.dataset_generation import (
    _create_prompts_for_generating_samples,
)
from testing.factories import vectors as vector_factories
from testing.helpers import zenml as zenml_helpers


@pytest.mark.parametrize(
    "dataset_type",
    [
        dataset_generation.DatasetType.INSTRUCT,
        dataset_generation.DatasetType.PREFERENCE,
    ],
)
def test_creates_prompts_for_generating_instruct_samples(
    dataset_type: dataset_generation.DatasetType,
):
    article = vector_factories.ArticleChunk()
    other_article = vector_factories.ArticleChunk()
    repository = vector_factories.RepositoryChunk()

    context = zenml_helpers.FakeContext()

    prompts = _create_prompts_for_generating_samples.create_prompts_for_generating_samples.entrypoint(
        documents=[article, other_article, repository],
        dataset_type=dataset_type,
        context=context,
    )

    assert len(prompts) == 3
    for prompt in prompts:
        assert prompt.render()

    assert context.output_metadata["generate_sample_prompts"]["num_prompts"] == 3
