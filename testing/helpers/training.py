from llm_twin.domain import training
from testing.factories import dataset as dataset_factories


class FakeInstructDatasetLoader(training.DataLoader):
    def load(self) -> dict:
        samples = [
            dataset_factories.InstructSample(),
            dataset_factories.InstructSample(),
        ]
        dataset = dataset_factories.SampleDataset(samples=samples)
        split_dataset = dataset.train_test_split(test_size=0.5)
        return {
            "train": split_dataset.train.serialize_for_hugging_face(),
            "test": split_dataset.train.serialize_for_hugging_face(),
        }


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
