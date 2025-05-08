from llm_twin.orchestration.steps.evaluation import _generate_answers
from testing.factories import dataset as dataset_factories
from testing.factories import documents as document_factories
from testing.helpers import config as config_helpers
from testing.helpers import storage as storage_helpers
from testing.helpers import zenml as zenml_helpers


def test_generates_answer_for_instruct_dataset_containing_single_sample():
    author = document_factories.Author()
    sample = dataset_factories.InstructSample(instruction="a")
    test_dataset = dataset_factories.InstructSampleDataset(samples=[sample])
    dataset = dataset_factories.InstructTrainTestSplit(
        author_id=author.id, test=test_dataset
    )

    context = zenml_helpers.FakeContext()
    db = storage_helpers.InMemoryVectorDatabase(vectors=[dataset])

    with config_helpers.install_in_memory_vector_db(db=db):
        pairs = _generate_answers.generate_answers.entrypoint(
            author_id=author.id,
            load_model_from="llamafactory/tiny-random-Llama-3",
            max_tokens=3,
            context=context,
        )

    assert len(dataset.test.samples) == len(pairs) == 1
    pair = pairs[0]
    assert pair.instruction == dataset.test.samples[0].instruction
    assert pair.answer == "a Anchor" # Random but seeded.

    assert context.output_metadata == {"answers": {"num_pairs": 1}}
