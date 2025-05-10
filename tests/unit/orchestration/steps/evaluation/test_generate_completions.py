from llm_twin.domain import training
from llm_twin.orchestration.steps.evaluation import _generate_completions
from testing.factories import dataset as dataset_factories
from testing.factories import documents as document_factories
from testing.helpers import config as config_helpers
from testing.helpers import storage as storage_helpers
from testing.helpers import zenml as zenml_helpers


def test_generates_completions_for_instruct_dataset_containing_single_sample():
    author = document_factories.Author()
    sample = dataset_factories.InstructSample(instruction="fixed")
    test_dataset = dataset_factories.InstructSampleDataset(samples=[sample])
    dataset = dataset_factories.InstructTrainTestSplit(
        author_id=author.id, test=test_dataset
    )

    context = zenml_helpers.FakeContext()
    db = storage_helpers.InMemoryVectorDatabase(vectors=[dataset])

    with config_helpers.install_in_memory_vector_db(db=db):
        completions = (
            _generate_completions.generate_completions_for_test_samples.entrypoint(
                author_id=author.id,
                load_model_from=training.BaseModelName.TINY_RANDOM.value,
                max_tokens=30,
                context=context,
            )
        )

    assert len(dataset.test.samples) == len(completions) == 1
    completion = completions[0]
    expected_prompt = training.render_alpaca_template(
        dataset.test.samples[0].instruction
    )
    assert completion.prompt == expected_prompt
    # Random but seeded.
    assert (
        completion.response
        == "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nfixed\n\n### Response:\n(CloneDispatchToProps Stitch"
    )

    assert context.output_metadata == {"completions": {"num_completions": 1}}
