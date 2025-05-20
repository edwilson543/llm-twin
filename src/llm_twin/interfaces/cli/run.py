import datetime as dt
import enum
import pathlib

import click

from llm_twin.domain import dataset_generation, training
from llm_twin.interfaces.cli import _exceptions
from llm_twin.orchestration import pipelines


class Pipeline(enum.StrEnum):
    ETL = "etl"
    FEATURE_ENGINEERING = "fe"
    DATASET_GENERATION = "dg"
    TRAINING = "train"
    EVALUATION = "eval"


@click.command(help="Entrypoint for running the ZenML pipeline", name="etl")
@click.option(
    "--disable-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--stop-after",
    type=click.Choice(choices=[pipeline.value for pipeline in Pipeline]),
    help="The pipeline to stop execution after.",
    default=Pipeline.EVALUATION.value,
)
@click.option(
    "--author",
    help="Name of the directory containing the configuration files for the particular author.",
    type=click.Choice(choices=["ed-wilson", "jackof-alltrades"]),
)
@click.option(
    "--base-model",
    help="Path to the base model that will be trained on the author's data.",
    type=click.Choice(choices=[model.value for model in training.BaseModelName]),
    default=training.BaseModelName.TINY_RANDOM.value,
)
def main(author: str, base_model: str, stop_after: str, disable_cache: bool) -> None:
    """
    Chain all the pipelines together into one massive pipeline.
    """
    stop_after_pipeline = Pipeline(stop_after)

    # ETL.
    etl_pipeline = pipelines.etl_author_data.with_options(
        run_name=f"etl-author-data-{_get_unique_pipeline_suffix()}",
        config_path=_get_config_file(author=author, pipeline="etl"),
        enable_cache=not disable_cache,
    )
    etl_pipeline_result = etl_pipeline()

    if stop_after_pipeline is Pipeline.ETL:
        return

    assert etl_pipeline_result is not None
    author_document = (
        etl_pipeline_result.steps["get_or_create_author"].outputs["author"][0].load()
    )

    # Feature engineering.
    feature_engineering_pipeline = (
        pipelines.process_raw_documents_into_features.with_options(
            run_name=f"feature-engineering-{_get_unique_pipeline_suffix()}",
            enable_cache=not disable_cache,
        )
    )
    feature_engineering_pipeline(author_full_name=author_document.full_name)
    if stop_after_pipeline is Pipeline.FEATURE_ENGINEERING:
        return

    # Instruct sample dataset generation.
    instruct_dataset_generation_pipeline = pipelines.generate_sample_dataset.with_options(
        run_name=f"generate-instruct-sample-dataset-{_get_unique_pipeline_suffix()}",
        enable_cache=not disable_cache,
    )
    instruct_dataset_generation_pipeline(
        author_id=author_document.id,
        dataset_type=dataset_generation.DatasetType.INSTRUCT,
        test_size=0.9,
    )

    # Preference sample dataset generation.
    preference_dataset_generation_pipeline = pipelines.generate_sample_dataset.with_options(
        run_name=f"generate-preference-sample-dataset-{_get_unique_pipeline_suffix()}",
        enable_cache=not disable_cache,
    )
    preference_dataset_generation_pipeline(
        author_id=author_document.id,
        dataset_type=dataset_generation.DatasetType.PREFERENCE,
        test_size=0.9,
    )

    if stop_after_pipeline is Pipeline.DATASET_GENERATION:
        return

    # Training.
    model_output_dir = pathlib.Path("/tmp/llm-twin/")
    training_pipeline = pipelines.train.with_options(
        run_name=f"train-model-{_get_unique_pipeline_suffix()}",
        enable_cache=not disable_cache,
    )
    training_pipeline(
        author_id=author_document.id,
        base_model_name=base_model,
        output_dir=model_output_dir,
        num_train_epochs=5,
        report_to="comet_ml",
    )
    if stop_after_pipeline is Pipeline.TRAINING:
        return

    # Evaluation.
    evaluation_pipeline = pipelines.evaluate_model.with_options(
        run_name=f"evaluate-model-{_get_unique_pipeline_suffix()}",
        enable_cache=not disable_cache,
    )
    evaluation_pipeline(
        author_id=author_document.id,
        load_model_from=str(model_output_dir / "dpo"),
        max_tokens=100,
    )
    if stop_after_pipeline is Pipeline.EVALUATION:
        return


def _get_config_file(*, author: str, pipeline: str) -> str:
    config_dir = pathlib.Path(__file__).parent / "config-files" / author
    config_path = config_dir / f"{pipeline}.yaml"

    if not config_path.is_file():
        raise _exceptions.ConfigFileDoesNotExist(filepath=config_path)

    return str(config_path)


def _get_unique_pipeline_suffix() -> str:
    return str(dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))


if __name__ == "__main__":
    main()
