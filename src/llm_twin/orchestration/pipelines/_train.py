import pathlib

import zenml

from llm_twin import config
from llm_twin.domain import training
from llm_twin.orchestration.steps import training as training_steps


@zenml.pipeline
def train(
    author_id: str,
    base_model_name: training.BaseModelName,
    num_train_epochs: int,
    report_to: str | None,
    output_dir: pathlib.Path | None = None,
) -> None:
    output_dir = output_dir or config.get_training_output_dir()

    sft_model_path = output_dir / "sft"
    training_steps.run_supervised_fine_tuning(
        author_id=author_id,
        base_model_name=base_model_name,
        load_model_from=base_model_name,
        output_dir=sft_model_path,
        num_train_epochs=num_train_epochs,
        report_to=report_to,
    )

    dpo_model_path = output_dir / "dpo"
    training_steps.run_direct_preference_optimisation(
        after="run_supervised_fine_tuning",
        author_id=author_id,
        base_model_name=base_model_name,
        load_model_from=str(sft_model_path),
        output_dir=dpo_model_path,
        num_train_epochs=num_train_epochs,
        report_to=report_to,
    )


if __name__ == "__main__":
    base_model_name = training.BaseModelName.TINY_RANDOM.value

    train.with_options(enable_cache=False)(
        author_id="29eec0df-0405-45a9-a3ef-40c314e15789",
        base_model_name=base_model_name,
        output_dir=pathlib.Path("/tmp/llm-twin/"),
        num_train_epochs=3,
        report_to="comet_ml",
    )
