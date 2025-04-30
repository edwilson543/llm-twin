import pathlib
import typing

import zenml

from llm_twin import config
from llm_twin.domain import training


@zenml.step
def run_supervised_fine_tuning(
    model_name_or_path: str,
    num_train_epochs: int,
    output_dir: pathlib.Path,
    report_to: str | None,
) -> typing.Annotated[str, "sft_model_path"]:
    db = config.get_vector_database()
    data_loader = training.VectorDBDataLoader(db=db)

    strategy = training.SupervisedFineTuning(
        data_loader=data_loader,
        model_name=model_name_or_path,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        report_to=report_to,
        optimizer="adamw_torch",
    )

    strategy.fine_tune()

    return str(output_dir)
