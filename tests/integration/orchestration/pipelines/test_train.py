import torch
import transformers

from llm_twin.domain import training
from llm_twin.orchestration.pipelines import _train
from testing.factories import dataset as dataset_factories
from testing.factories import documents as document_factories


def test_trains_model_using_sft_then_dpo(output_dir):
    author = document_factories.Author.build()  # In memory is fine here.
    dataset_factories.InstructTrainTestSplit.create(author_id=author.id)
    dataset_factories.PreferenceTrainTestSplit.create(author_id=author.id)

    _train.train.entrypoint(
        author_id=author.id,
        base_model_name=training.BaseModelName.TINY_RANDOM,
        output_dir=output_dir,
        num_train_epochs=2,
        report_to=None,
    )

    # Load the tuned model, and use it to generate some dummy outputs.
    dpo_trained_model = transformers.AutoModelForCausalLM.from_pretrained(
        output_dir / "dpo"
    )

    dummy_output_tokens = dpo_trained_model.generate(max_length=2, top_k=1)
    assert dummy_output_tokens.equal(torch.tensor([[128000, 28510]]))
