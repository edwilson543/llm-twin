from testing.factories import dataset as dataset_factories


class TestDataset:
    def test_splits_dataset_of_two_samples_down_the_middle(self):
        sample = dataset_factories.InstructSample()
        other_sample = dataset_factories.InstructSample()

        dataset = dataset_factories.InstructSampleDataset(
            samples=[sample, other_sample],
        )

        train_test_split = dataset.train_test_split(test_size=0.5, random_state=31)

        assert train_test_split.train.samples == [sample]
        assert train_test_split.test.samples == [other_sample]
        assert train_test_split.test_size == 0.5
