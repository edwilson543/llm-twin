import pathlib

import zenml

from llm_twin.orchestration.steps import training as training_steps


@zenml.pipeline
def train_locally(
    base_model_name: str,
    output_dir: pathlib.Path,
    num_train_epochs: int,
    report_to: str | None,
) -> None:
    sft_model_path = training_steps.run_supervised_fine_tuning_locally(
        model_name_or_path=base_model_name,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        report_to=report_to,
    )

    training_steps.run_direct_preference_optimisation_locally(
        model_name_or_path=str(sft_model_path),
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        report_to=report_to,
    )
