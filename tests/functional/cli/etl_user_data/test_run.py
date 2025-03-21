import pytest
from click import testing as click_testing
from zenml import client as zenml_client

from llm_twin import settings
from llm_twin.domain import documents
from llm_twin.interfaces.cli import exceptions as cli_exceptions
from llm_twin.interfaces.cli.etl_user_data.run import run


def test_runs_etl_pipeline_and_persists_outcome():
    args = ["--config-filename", "jackof-alltrades.yaml", "--disable-cache"]
    command_runner = click_testing.CliRunner()

    result = command_runner.invoke(run, args)

    assert result.exit_code == 0

    # Ensure the relevant articles were extracted.
    db = settings.get_nosql_database()
    author = documents.UserDocument.get(
        db=db, first_name="Jackof", last_name="Alltrades"
    )

    first_post = documents.ArticleDocument.get(
        db=db, link="https://fake.com/blog/ten-things-to-be-average-at.html"
    )
    assert first_post.platform == "fake"
    assert first_post.author_id == author.id

    second_post = documents.ArticleDocument.get(
        db=db, link="https://fake.com/blog/top-tips-for-mediocracy.html"
    )
    assert second_post.platform == "fake"
    assert second_post.author_id == author.id

    # Ensure the pipeline was run end-to-end and completed successfully.
    pipeline_run = zenml_client.Client().list_pipeline_runs(
        pipeline="etl_user_data", sort_by="desc:created"
    )[0]
    assert pipeline_run.status == "completed"


def test_raises_when_config_file_does_not_exist():
    args = ["--config-filename", "does-not-exist.yaml", "--disable-cache"]
    command_runner = click_testing.CliRunner()

    with pytest.raises(cli_exceptions.ConfigFileDoesNotExist) as exc:
        command_runner.invoke(run, args, catch_exceptions=False)

    assert "does-not-exist.yaml" in str(exc.value.filepath)
