from llm_twin.domain import dataset_generation, training
from testing.factories import dataset as dataset_factories


class FakeInstructDatasetLoader(training.DataLoader):
    def load(self) -> dataset_generation.TrainTestSplit:
        samples = [
            dataset_factories.InstructSample(),
            dataset_factories.InstructSample(),
        ]
        dataset = dataset_factories.SampleDataset.build(samples=samples)
        return dataset.train_test_split(test_size=0.5)
