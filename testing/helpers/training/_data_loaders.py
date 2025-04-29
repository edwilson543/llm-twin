from llm_twin.domain import dataset_generation, training
from testing.factories import dataset as dataset_factories


class FakeDatasetLoader(training.DataLoader):
    def load_instruct_dataset(
        self,
    ) -> dataset_generation.TrainTestSplit[dataset_generation.InstructSample]:
        samples = [
            dataset_factories.InstructSample(),
            dataset_factories.InstructSample(),
        ]
        dataset = dataset_factories.SampleDataset.build(samples=samples)
        return dataset.train_test_split(test_size=0.5)

    def load_preference_dataset(
        self,
    ) -> dataset_generation.TrainTestSplit[dataset_generation.PreferenceSample]:
        samples = [
            dataset_factories.PreferenceSample(),
            dataset_factories.PreferenceSample(),
        ]
        dataset = dataset_factories.SampleDataset.build(samples=samples)
        return dataset.train_test_split(test_size=0.5)
