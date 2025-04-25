from llm_twin.domain import training
from llm_twin.orchestration.steps.training import _train
from testing.helpers import config as config_helpers
from testing.helpers import zenml as zenml_helpers


def test_uses_fake_trainer_to_train_model() -> None:
    context = zenml_helpers.FakeContext()

    with config_helpers.install_fake_trainer():
        _train.train.entrypoint(
            fine_tune_type=training.FineTuneType.SUPERVISED_FINE_TUNING,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            learning_rate=0.1,
            context=context,
        )
