import torch
import transformers

from llm_twin.orchestration.pipelines import _train
from testing.factories import dataset as dataset_factories


def test_trains_model_using_sft_then_dpo(output_dir):
    base_model_name = "llamafactory/tiny-random-Llama-3"

    dataset_factories.InstructTrainTestSplit.create()
    dataset_factories.PreferenceTrainTestSplit.create()

    _train.train.entrypoint(
        base_model_name=base_model_name,
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
