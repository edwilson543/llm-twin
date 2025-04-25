import dataclasses
from pathlib import Path

from huggingface_hub import HfApi
from loguru import logger
from sagemaker import huggingface as sagemaker_huggingface

from . import _base


finetuning_dir = Path(__file__).resolve().parent
finetuning_requirements_path = finetuning_dir / "requirements.txt"


@dataclasses.dataclass(frozen=True)
class SageMaker(_base.Trainer):
    """
    Run fine-tuning on AWS SageMaker.
    """

    _aws_role_arn: str
    _comet_api_key: str
    _comet_project_name: str
    _hugging_face_access_token: str
    _huggingface_dataset_workspace: str

    def run_fine_tuning(
        self,
        *,
        fine_tune_type: _base.FineTuneType,
        num_train_epochs: int,
        per_device_train_batch_size: int,
        learning_rate: float,
    ) -> None:
        if not finetuning_dir.exists():
            raise _base.UnableToRunFineTuning(
                f"The directory {finetuning_dir} does not exist."
            )
        if not finetuning_requirements_path.exists():
            raise _base.UnableToRunFineTuning(
                f"The file {finetuning_requirements_path} does not exist."
            )

        api = HfApi()
        user_info = api.whoami(token=self._hugging_face_access_token)
        huggingface_user = user_info["name"]
        logger.info(f"Current Hugging Face user: {huggingface_user}")

        hyperparameters = {
            "fine_tune_type": fine_tune_type.value,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "learning_rate": learning_rate,
            "dataset_huggingface_workspace": self._huggingface_dataset_workspace,
            "model_output_huggingface_workspace": huggingface_user,
        }

        huggingface_estimator = sagemaker_huggingface.HuggingFace(
            entry_point="_finetune.py",
            source_dir=str(finetuning_dir),
            instance_type="ml.g5.2xlarge",
            instance_count=1,
            role=self._aws_role_arn,
            transformers_version="4.36",
            pytorch_version="2.1",
            py_version="py310",
            hyperparameters=hyperparameters,
            requirements_file=finetuning_requirements_path,
            environment={
                "HUGGING_FACE_HUB_TOKEN": self._hugging_face_access_token,
                "COMET_API_KEY": self._comet_api_key,
                "COMET_PROJECT_NAME": self._comet_project_name,
            },
        )

        huggingface_estimator.fit()
