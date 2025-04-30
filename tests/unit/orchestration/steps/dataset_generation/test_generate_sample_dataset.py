from llm_twin.domain import dataset_generation
from llm_twin.orchestration.steps.dataset_generation import _generate_sample_dataset
from testing.factories import dataset as dataset_factories
from testing.helpers import config as config_helpers
from testing.helpers import zenml as zenml_helpers


def test_generates_and_splits_instruct_sample_dataset():
    prompt = dataset_factories.GenerateInstructSamplePrompt()
    other_prompt = dataset_factories.GenerateInstructSamplePrompt()

    context = zenml_helpers.FakeContext()

    with (
        config_helpers.install_in_memory_vector_db() as db,
        config_helpers.install_fake_language_model(),
    ):
        dataset = _generate_sample_dataset.generate_sample_dataset.entrypoint(
            dataset_type=dataset_generation.DatasetType.INSTRUCT,
            prompts=[prompt, other_prompt],
            test_size=0.5,
            context=context,
        )

    assert db.vectors == [dataset]
    assert dataset.dataset_type == dataset_generation.DatasetType.INSTRUCT

    assert len(dataset.train.samples) == 1 * dataset_factories.SAMPLES_PER_PROMPT
    train_sample = dataset.train.samples[0]
    assert isinstance(train_sample, dataset_generation.InstructSample)

    assert len(dataset.test.samples) == 1 * dataset_factories.SAMPLES_PER_PROMPT
    test_sample = dataset.test.samples[0]
    assert isinstance(test_sample, dataset_generation.InstructSample)

    assert context.output_metadata["sample_dataset"] == {
        "dataset_type": "INSTRUCT",
        "num_samples": 10,
    }


def test_generates_and_splits_preference_sample_dataset():
    prompts = [dataset_factories.GeneratePreferenceSamplePrompt() for _ in range(4)]

    context = zenml_helpers.FakeContext()

    with (
        config_helpers.install_in_memory_vector_db() as db,
        config_helpers.install_fake_language_model(),
    ):
        dataset = _generate_sample_dataset.generate_sample_dataset.entrypoint(
            dataset_type=dataset_generation.DatasetType.PREFERENCE,
            prompts=prompts,
            test_size=0.25,
            context=context,
        )

    assert db.vectors == [dataset]
    assert dataset.dataset_type == dataset_generation.DatasetType.PREFERENCE

    assert len(dataset.train.samples) == 3 * dataset_factories.SAMPLES_PER_PROMPT
    for train_sample in dataset.train.samples:
        assert isinstance(train_sample, dataset_generation.PreferenceSample)

    assert len(dataset.test.samples) == 1 * dataset_factories.SAMPLES_PER_PROMPT
    test_sample = dataset.test.samples[0]
    assert isinstance(test_sample, dataset_generation.PreferenceSample)

    assert context.output_metadata["sample_dataset"] == {
        "dataset_type": "PREFERENCE",
        "num_samples": 20,
    }
