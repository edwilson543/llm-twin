from llm_twin.domain import dataset_generation, training


class TestHuggingFaceDataLoader:
    def test_can_load_mlabonne_instruct_dataset(self):
        data_loader = training.HuggingFaceDataLoader(
            dataset_path="mlabonne/FineTome-Alpaca-100k",
            split="train[:10]",
            test_size=0.1,
        )

        dataset = data_loader.load_instruct_dataset(author_id="mlabonne")

        assert dataset.author_id == "mlabonne"
        assert dataset.dataset_type == dataset_generation.DatasetType.INSTRUCT
        assert all(
            isinstance(sample, dataset_generation.InstructSample)
            for sample in dataset.all_samples
        )
        assert len(dataset.train.samples) == 9
        assert len(dataset.test.samples) == 1

    def test_can_load_mlabonne_preference_dataset(self):
        data_loader = training.HuggingFaceDataLoader(
            dataset_path="mlabonne/llmtwin-dpo",
            split="train[:10]",
            test_size=0.1,
        )

        dataset = data_loader.load_preference_dataset(author_id="mlabonne")

        assert dataset.author_id == "mlabonne"
        assert dataset.dataset_type == dataset_generation.DatasetType.PREFERENCE
        assert all(
            isinstance(sample, dataset_generation.PreferenceSample)
            for sample in dataset.all_samples
        )
        assert len(dataset.train.samples) == 9
        assert len(dataset.test.samples) == 1
