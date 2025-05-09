from llm_twin.orchestration.pipelines import _evaluation
from testing.factories import dataset as dataset_factories
from testing.helpers import config as config_helpers


def test_evaluates_model_using_instruct_samples():
    sample = dataset_factories.InstructSample(instruction="fixed")
    test_dataset = dataset_factories.InstructSampleDataset(samples=[sample])
    dataset = dataset_factories.InstructTrainTestSplit.create(test=test_dataset)

    with config_helpers.install_fake_language_model():
        _evaluation.evaluate_model.entrypoint(
            author_id=dataset.author_id,
            load_model_from="llamafactory/tiny-random-Llama-3",
            max_tokens=3,
        )
