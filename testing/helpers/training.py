from llm_twin.domain import training, dataset_generation
from testing.factories import dataset as dataset_factories


class FakeInstructDatasetLoader(training.DataLoader):
    def load(self) -> dataset_generation.TrainTestSplit:
        samples = [
            dataset_factories.InstructSample(),
            dataset_factories.InstructSample(),
        ]
        dataset = dataset_factories.SampleDataset(samples=samples)
        return dataset.train_test_split(test_size=0.5)


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
