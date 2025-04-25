import zenml

from llm_twin.domain import training
from llm_twin.orchestration.steps import training as training_steps


@zenml.pipeline
def training_pipeline(
    fine_tune_type: training.FineTuneType,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 3e-4,
) -> None:
    training_steps.train(
        fine_tune_type=fine_tune_type,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
    )
