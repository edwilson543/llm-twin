import pathlib

import zenml

from llm_twin import config
from llm_twin.domain import training

from . import _reporting


@zenml.step
def run_direct_preference_optimisation(
    base_model_name: str,
    load_model_from: str,
    num_train_epochs: int,
    output_dir: pathlib.Path,
    report_to: str | None,
) -> None:
    db = config.get_vector_database()
    data_loader = training.VectorDBDataLoader(db=db)

    strategy = training.DirectPreferenceOptimisation(
        data_loader=data_loader,
        model_name=load_model_from,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        report_to=report_to,
        optimizer="adamw_torch",
    )

    with _reporting.create_training_report(name=f"dpo:{base_model_name}", report_to=report_to):
        strategy.fine_tune()
