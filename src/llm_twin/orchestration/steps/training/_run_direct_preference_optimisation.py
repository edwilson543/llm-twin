import pathlib

import zenml

from llm_twin import config
from llm_twin.domain import training

from . import _reporting


@zenml.step
def run_direct_preference_optimisation(
    author_id: str,
    base_model_name: training.BaseModelName,
    load_model_from: str,
    num_train_epochs: int,
    output_dir: pathlib.Path,
    report_to: str | None,
) -> None:
    db = config.get_vector_database()
    data_loader = training.VectorDBDataLoader(db=db)
    dataset = data_loader.load_preference_dataset(author_id=author_id)

    strategy = training.DirectPreferenceOptimisation(
        model_name=load_model_from,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        report_to=report_to,
        optimizer="adamw_torch",
    )

    with _reporting.create_training_report(
        name=f"dpo:{base_model_name.value}", report_to=report_to
    ):
        strategy.fine_tune(dataset=dataset)
