import zenml

from llm_twin import config
from llm_twin.domain import training
from llm_twin.orchestration.steps import context


@zenml.step
def train(
    fine_tune_type: training.FineTuneType,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    learning_rate: float,
    context: context.StepContext | None = None,
) -> None:
    trainer = config.get_trainer()

    trainer.run_fine_tuning(
        fine_tune_type=fine_tune_type,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
    )

    del context
