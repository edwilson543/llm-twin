import pathlib
import shutil
import typing
import uuid

import pytest
import torch
import transformers

from llm_twin.domain.training._fine_tuning_strategy import _supervised_fine_tuning
from testing.helpers import training as training_helpers


class TestFineTune:
    @pytest.fixture
    def output_dir(self) -> typing.Generator[pathlib.Path, None, None]:
        output_dir = pathlib.Path(__file__).parent / "outputs" / str(uuid.uuid4())
        output_dir.mkdir(parents=True, exist_ok=False)

        try:
            yield output_dir
        finally:
            shutil.rmtree(output_dir)

    def test_can_fine_tune_tiny_random_llama_on_fake_dataset(self, output_dir):
        model_name = "llamafactory/tiny-random-Llama-3"
        data_loader = training_helpers.FakeDatasetLoader()

        strategy = _supervised_fine_tuning.SupervisedFineTuning(
            data_loader=data_loader,
            model_name=model_name,
            output_dir=output_dir,
            num_train_epochs=1,
            report_to=None, # Don't report training data anywhere.
            optimizer="adamw_torch",  # `adamw_8bit` not available on MacOS.
        )

        strategy.fine_tune()

        # Load the tuned model, and use it to generate some dummy outputs.
        model = transformers.AutoModelForCausalLM.from_pretrained(output_dir)

        dummy_output_tokens = model.generate(max_length=2, top_k=1)
        assert dummy_output_tokens.equal(torch.tensor([[128000, 28510]]))
