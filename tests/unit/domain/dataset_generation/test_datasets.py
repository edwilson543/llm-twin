from llm_twin.domain.dataset_generation import _datasets
from llm_twin.domain.storage import vector as vector_storage
from testing.factories import dataset as dataset_factories


class TestDataset:
    def test_splits_dataset_of_two_samples_down_the_middle(self):
        sample = dataset_factories.InstructSample()
        other_sample = dataset_factories.InstructSample()

        dataset = _datasets.SampleDataset(
            input_data_category=vector_storage.DataCategory.TESTING,
            samples=[sample, other_sample],
        )

        train, test = dataset.train_test_split(test_size=0.5, random_state=31)

        assert train.input_data_category == dataset.input_data_category
        assert train.samples == [sample]

        assert test.input_data_category == dataset.input_data_category
        assert test.samples == [other_sample]
