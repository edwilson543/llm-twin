from llm_twin.domain.dataset_generation import _datasets, _generation
from testing.factories import dataset as dataset_factories
from testing.helpers import models as models_helpers


class TestGenerateSampleDataset:
    def test_generates_instruct_sample_dataset_from_prompts(self):
        language_model = models_helpers.FakeLanguageModel()
        system_prompt = dataset_factories.Prompt()

        author_id = "edwilson543"
        prompt = dataset_factories.GenerateInstructSamplePrompt()
        other_prompt = dataset_factories.GenerateInstructSamplePrompt()

        dataset = _generation.generate_sample_dataset(
            author_id=author_id,
            dataset_type=_datasets.DatasetType.INSTRUCT,
            language_model=language_model,
            system_prompt=system_prompt,
            prompts=[prompt, other_prompt],
        )

        assert dataset.author_id == author_id
        assert dataset.dataset_type == _datasets.DatasetType.INSTRUCT
        assert dataset.num_samples == 2 * dataset_factories.SAMPLES_PER_PROMPT
        assert all(
            isinstance(sample, _datasets.InstructSample) for sample in dataset.samples
        )

    def test_generates_preference_sample_dataset_from_prompts(self):
        language_model = models_helpers.FakeLanguageModel()
        system_prompt = dataset_factories.Prompt()

        author_id = "edwilson543"
        prompt = dataset_factories.GeneratePreferenceSamplePrompt()

        dataset = _generation.generate_sample_dataset(
            author_id=author_id,
            dataset_type=_datasets.DatasetType.PREFERENCE,
            language_model=language_model,
            system_prompt=system_prompt,
            prompts=[prompt],
        )

        assert dataset.author_id == author_id
        assert dataset.dataset_type == _datasets.DatasetType.PREFERENCE
        assert dataset.num_samples == 1 * dataset_factories.SAMPLES_PER_PROMPT
        assert all(
            isinstance(sample, _datasets.PreferenceSample) for sample in dataset.samples
        )
