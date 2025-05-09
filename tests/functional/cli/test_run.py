from click import testing as click_testing
from zenml import client as zenml_client

from llm_twin import config
from llm_twin.domain import authors, dataset_generation
from llm_twin.domain.etl import raw_documents
from llm_twin.domain.feature_engineering import embedding
from llm_twin.interfaces.cli import run
from testing.helpers import config as config_helpers


def test_raises_when_invalid_author_option_given():
    args = ["--author", "does-not-exist", "--disable-cache"]
    command_runner = click_testing.CliRunner()

    result = command_runner.invoke(
        run, args, catch_exceptions=False, prog_name="e2e-test"
    )

    assert result.exit_code == 2
    expected_error = "Error: Invalid value for '--author': 'does-not-exist' is not one of 'ed-wilson', 'jackof-alltrades'."
    assert expected_error in result.output


def test_runs_all_pipelines_back_to_back():
    args = ["--author", "jackof-alltrades", "--stop-after", "dg", "--disable-cache"]
    command_runner = click_testing.CliRunner()

    with config_helpers.install_fake_language_model():
        result = command_runner.invoke(run, args, prog_name="e2e-test")

    assert result.exit_code == 0
    _make_etl_assertions()
    _make_feature_engineering_assertions()
    _make_instruct_sample_dataset_assertions()
    _make_preference_sample_dataset_assertions()


def _make_etl_assertions() -> None:
    pipeline_run = zenml_client.Client().list_pipeline_runs(
        pipeline="etl_author_data", sort_by="desc:created"
    )[0]
    assert pipeline_run.status == "completed"

    # Ensure the relevant articles were extracted.
    db = config.get_document_database()
    author = db.find_one(
        document_class=authors.Author, first_name="Jackof", last_name="Alltrades"
    )

    first_post = db.find_one(
        document_class=raw_documents.Article,
        link="https://fake.com/blog/ten-things-to-be-average-at.html",
    )
    assert first_post.platform == "fake"
    assert first_post.author_id == author.id

    second_post = db.find_one(
        document_class=raw_documents.Article,
        link="https://fake.com/blog/top-tips-for-mediocracy.html",
    )
    assert second_post.platform == "fake"
    assert second_post.author_id == author.id


def _make_feature_engineering_assertions() -> None:
    pipeline_run = zenml_client.Client().list_pipeline_runs(
        pipeline="process_raw_documents_into_features", sort_by="desc:created"
    )[0]
    assert pipeline_run.status == "completed"

    db = config.get_vector_database()
    embedded_articles, next_offset = db.bulk_find(
        vector_class=embedding.EmbeddedArticleChunk, limit=5
    )
    assert next_offset is None
    assert len(embedded_articles) == 2


def _make_instruct_sample_dataset_assertions() -> None:
    pipeline_run = zenml_client.Client().list_pipeline_runs(
        pipeline="generate_sample_dataset", sort_by="desc:created"
    )[1]  # `1` since it's run before the preference dataset generation pipeline.
    assert pipeline_run.status == "completed"

    db = config.get_vector_database()
    datasets, next_offset = db.bulk_find(
        vector_class=dataset_generation.TrainTestSplit,
        dataset_type=dataset_generation.DatasetType.INSTRUCT.value,
        limit=1,
    )
    assert next_offset is None
    instruct_dataset = datasets[0]
    assert len(instruct_dataset.all_samples) == 10


def _make_preference_sample_dataset_assertions() -> None:
    pipeline_run = zenml_client.Client().list_pipeline_runs(
        pipeline="generate_sample_dataset", sort_by="desc:created"
    )[0]
    assert pipeline_run.status == "completed"

    db = config.get_vector_database()
    datasets, next_offset = db.bulk_find(
        vector_class=dataset_generation.TrainTestSplit,
        dataset_type=dataset_generation.DatasetType.PREFERENCE.value,
        limit=1,
    )
    assert next_offset is None
    preference_dataset = datasets[0]
    assert len(preference_dataset.all_samples) == 10
