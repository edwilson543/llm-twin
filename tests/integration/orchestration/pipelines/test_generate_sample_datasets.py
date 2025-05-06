import random

import pytest

from llm_twin.config._settings import settings
from llm_twin.domain import dataset_generation
from llm_twin.orchestration.pipelines import _dataset_generation
from testing.factories import documents as document_factories
from testing.factories import vectors as vector_factories
from testing.helpers import config as config_helpers
from testing.helpers import zenml as zenml_helpers


def test_generates_sample_dataset_using_fake_language_model():
    author = document_factories.Author.create()

    vector_factories.ArticleChunk.create(author=author)
    vector_factories.ArticleChunk.create(author=author)
    vector_factories.RepositoryChunk.create(author=author)

    # Create an article chunk by some other author, that samples shouldn't be generated from.
    other_author = document_factories.Author.create()
    vector_factories.ArticleChunk.create(author=other_author)

    pipeline = _dataset_generation.generate_sample_dataset.with_options(
        enable_cache=False
    )
    with config_helpers.install_fake_language_model():
        pipeline.entrypoint(
            author_id=author.id,
            dataset_type=dataset_generation.DatasetType.PREFERENCE,
            test_size=0.2,
        )

    sample_dataset = zenml_helpers.load_artifact_from_most_recent_pipeline_run(
        step_name="generate_sample_dataset", output_name="sample_dataset"
    )

    assert isinstance(sample_dataset, dataset_generation.TrainTestSplit)
    assert all(
        isinstance(sample, dataset_generation.PreferenceSample)
        for sample in sample_dataset.all_samples
    )
    assert len(sample_dataset.train.samples) == 12
    assert len(sample_dataset.test.samples) == 3


@pytest.mark.skipif(
    # Only run the test when the OpenAI API key is set (i.e. in local dev),
    # and event then, only 5% of the time.
    not settings.OPENAI_API_KEY,
    random.random() < 0.05,
    reason="Is slow and requires OpenAI API call",
)
def test_generates_sample_dataset_using_gpt():
    chunk = vector_factories.ArticleChunk.create(
        content="King Gizzard & the Lizard Wizard."
    )

    pipeline = _dataset_generation.generate_sample_dataset.with_options(
        enable_cache=False
    )
    pipeline.entrypoint(
        author_id=chunk.author_id,
        dataset_type=dataset_generation.DatasetType.INSTRUCT,
        test_size=0.2,
    )

    sample_dataset = zenml_helpers.load_artifact_from_most_recent_pipeline_run(
        step_name="generate_sample_dataset", output_name="sample_dataset"
    )

    assert isinstance(sample_dataset, dataset_generation.TrainTestSplit)
    assert all(
        isinstance(sample, dataset_generation.InstructSample)
        for sample in sample_dataset.all_samples
    )
    assert len(sample_dataset.train.samples) == 4
    assert len(sample_dataset.test.samples) == 1
