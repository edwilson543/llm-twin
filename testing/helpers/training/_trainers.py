from llm_twin.domain import training


class FakeTrainer(training.Trainer):
    def run_fine_tuning(
        self,
        *,
        fine_tune_type: training.FineTuneType,
        num_train_epochs: int,
        per_device_train_batch_size: int,
        learning_rate: float,
    ) -> None:
        return None
