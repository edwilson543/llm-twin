import torch
import transformers

from llm_twin.domain import training
from testing.factories import dataset as dataset_factories


class TestFineTune:
    def test_can_fine_tune_tiny_random_llama_on_fake_dataset(self, output_dir):
        dataset = dataset_factories.PreferenceTrainTestSplit()

        model_name = training.BaseModelName.TINY_RANDOM
        strategy = training.DirectPreferenceOptimisation(
            model_name=model_name.value,
            output_dir=output_dir,
            report_to=None,  # Don't report training data anywhere.
            num_train_epochs=1,
            optimizer="adamw_torch",  # `adamw_8bit` not available on MacOS.
        )

        strategy.fine_tune(dataset=dataset)

        # Load the tuned model, and use it to generate some dummy outputs.
        model = transformers.AutoModelForCausalLM.from_pretrained(output_dir)

        dummy_output_tokens = model.generate(max_length=2, top_k=1)
        assert dummy_output_tokens.equal(torch.tensor([[128000, 28510]]))
