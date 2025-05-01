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
        base_model_name=base_model_name,
        load_model_from=base_model_name,
        output_dir=sft_model_path,
        num_train_epochs=num_train_epochs,
        report_to=report_to,
    )

    dpo_model_path = output_dir / "dpo"
    training_steps.run_direct_preference_optimisation(
        base_model_name=base_model_name,
        load_model_from=str(sft_model_path),
        output_dir=dpo_model_path,
        num_train_epochs=num_train_epochs,
        report_to=report_to,
    )


if __name__ == "__main__":
    base_model_name = "llamafactory/tiny-random-Llama-3"

    train.with_options(enable_cache=False)(
        base_model_name=base_model_name,
        output_dir=pathlib.Path("/tmp/llm-twin/"),
        num_train_epochs=3,
        report_to="comet_ml",
    )
