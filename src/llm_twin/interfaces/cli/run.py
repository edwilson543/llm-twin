from datetime import datetime as dt
from pathlib import Path

import click

from llm_twin.domain import dataset_generation
from llm_twin.interfaces.cli import _exceptions
from llm_twin.orchestration import pipelines


@click.command(help="Entrypoint for running the ZenML pipeline", name="etl")
@click.option(
    "--disable-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--author",
    help="Name of the directory containing the configuration files for the particular author.",
    type=click.Choice(choices=["ed-wilson", "jackof-alltrades"]),
)
def main(author: str, disable_cache: bool) -> None:
    # ETL.
    etl_pipeline = pipelines.etl_author_data.with_options(
        run_name=f"etl-author-data-{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        config_path=_get_config_file(author=author, pipeline="etl"),
        enable_cache=not disable_cache,
    )

    etl_pipeline_result = etl_pipeline()
    assert etl_pipeline_result is not None
    author_document = (
        etl_pipeline_result.steps["get_or_create_author"].outputs["author"][0].load()
    )

    # Feature engineering.
    feature_engineering_pipeline = (
        pipelines.process_raw_documents_into_features.with_options(
            run_name=f"feature-engineering-{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}",
            enable_cache=not disable_cache,
        )
    )
    feature_engineering_pipeline(author_full_name=author_document.full_name)

    # Instruct sample dataset generation.
    instruct_dataset_generation_pipeline = pipelines.generate_sample_dataset.with_options(
        run_name=f"generate-instruct-sample-dataset-{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        enable_cache=not disable_cache,
    )
    instruct_dataset_generation_pipeline(
        author_id=author_document.id,
        dataset_type=dataset_generation.DatasetType.INSTRUCT,
        test_size=0.9,
    )

    # Preference sample dataset generation.
    preference_dataset_generation_pipeline = pipelines.generate_sample_dataset.with_options(
        run_name=f"generate-preference-sample-dataset-{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        enable_cache=not disable_cache,
    )
    preference_dataset_generation_pipeline(
        author_id=author_document.id,
        dataset_type=dataset_generation.DatasetType.PREFERENCE,
        test_size=0.9,
    )


def _get_config_file(*, author: str, pipeline: str) -> str:
    config_dir = Path(__file__).parent / "config-files" / author
    config_path = config_dir / f"{pipeline}.yaml"

    if not config_path.is_file():
        raise _exceptions.ConfigFileDoesNotExist(filepath=config_path)

    return str(config_path)


if __name__ == "__main__":
    main()
