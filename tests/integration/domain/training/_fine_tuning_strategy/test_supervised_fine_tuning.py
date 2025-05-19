from llm_twin.domain import inference, training
from testing.factories import dataset as dataset_factories


class TestFineTune:
    def test_can_fine_tune_tiny_random_llama_on_fake_dataset(self, output_dir):
        dataset = dataset_factories.InstructTrainTestSplit()

        model_name = model_name = training.BaseModelName.TINY_RANDOM
        strategy = training.SupervisedFineTuning(
            model_name=model_name.value,
            output_dir=output_dir,
            num_train_epochs=1,
            report_to=None,  # Don't report training data anywhere.
            optimizer="adamw_torch",  # `adamw_8bit` not available on MacOS.
        )

        strategy.fine_tune(dataset=dataset)

        # Load the tuned model, and use it to generate a dummy response.
        model = inference.LocalInferenceEngine(load_model_from=output_dir)
        _, dummy_output = model.get_response(
            instruction="Hello?", max_tokens=30, top_k=1
        )
        assert dummy_output.endswith("(CloneDispatchToProps Stitch")
