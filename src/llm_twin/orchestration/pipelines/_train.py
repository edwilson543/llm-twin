import pathlib

import zenml

from llm_twin.orchestration.steps import training as training_steps


@zenml.pipeline
def train(
    base_model_name: str,
    output_dir: pathlib.Path,
    num_train_epochs: int,
    report_to: str | None,
) -> None:
    sft_model_path = output_dir / "sft"
    training_steps.run_supervised_fine_tuning(
        model_name_or_path=base_model_name,
        output_dir=sft_model_path,
        num_train_epochs=num_train_epochs,
        report_to=report_to,
    )

    dpo_model_path = output_dir / "dpo"
    training_steps.run_direct_preference_optimisation(
        model_name_or_path=str(sft_model_path),
        output_dir=dpo_model_path,
        num_train_epochs=num_train_epochs,
        report_to=report_to,
    )
