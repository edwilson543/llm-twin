import pytest
from click import testing as click_testing
from zenml import client as zenml_client

from llm_twin import config
from llm_twin.domain import authors
from llm_twin.domain.etl import raw_documents
from llm_twin.interfaces.cli import _exceptions as cli_exceptions
from llm_twin.interfaces.cli import etl


def test_runs_etl_pipeline_and_persists_outcome():
    args = ["--config-filename", "jackof-alltrades.yaml", "--disable-cache"]
    command_runner = click_testing.CliRunner()

    result = command_runner.invoke(etl, args, prog_name="etl-test")

    assert result.exit_code == 0

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

    # Ensure the pipeline was run end-to-end and completed successfully.
    pipeline_run = zenml_client.Client().list_pipeline_runs(
        pipeline="etl_author_data", sort_by="desc:created"
    )[0]
    assert pipeline_run.status == "completed"


def test_raises_when_config_file_does_not_exist():
    args = ["--config-filename", "does-not-exist.yaml", "--disable-cache"]
    command_runner = click_testing.CliRunner()

    with pytest.raises(cli_exceptions.ConfigFileDoesNotExist) as exc:
        command_runner.invoke(etl, args, catch_exceptions=False, prog_name="etl-test")

    assert "does-not-exist.yaml" in str(exc.value.filepath)
