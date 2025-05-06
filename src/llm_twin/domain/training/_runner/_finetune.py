import argparse
import os
import pathlib

import loguru

from llm_twin.domain.training import _data_loader, _fine_tuning_strategy

from . import _base


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--dataset_huggingface_workspace", type=str, default="mlabonne")
    parser.add_argument(
        "--model_output_huggingface_workspace", type=str, default="mlabonne"
    )
    parser.add_argument(
        "--fine_tune_type",
        type=str,
        choices=[
            _base.FineTuneType.SUPERVISED_FINE_TUNING.value,
            _base.FineTuneType.DIRECT_PREFERENCE_OPTIMISATION.value,
        ],
        default="sft",
        help="Parameter to choose the fine tuning stage.",
    )

    parser.add_argument(
        "--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    return parser.parse_args()


def _run_supervised_fine_tuning(*, args: argparse.Namespace) -> None:
    loguru.logger.info("Starting SFT training...")

    base_model_name = "meta-llama/Llama-3.1-8B"
    loguru.logger.info(f"Training from base model '{base_model_name}'")

    output_dir_sft = pathlib.Path(args.model_dir) / "output_sft"

    data_loader = _data_loader.HuggingFaceDataLoader(
        dataset_path="mlabonne/FineTome-Alpaca-100k",
    )
    dataset = data_loader.load_instruct_dataset(author_id="")  # TODO

    tuner = _fine_tuning_strategy.SupervisedFineTuning(
        output_dir=output_dir_sft,
        model_name=base_model_name,
        report_to="comet_ml",
    )

    tuner.fine_tune(dataset=dataset)


if __name__ == "__main__":
    args = _parse_arguments()

    loguru.logger.info(
        "Fine tuning parameters",
        params={
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "learning_rate": args.learning_rate,
            "dataset_huggingface_workspace": args.dataset_huggingface_workspace,
            "model_output_huggingface_workspace": args.model_output_huggingface_workspace,
            "fine_tune_type": args.fine_tune_type,
            "output_data_dir": args.output_data_dir,
            "model_dir": args.model_dir,
            "n_gpus": args.n_gpus,
        },
    )

    if args.fine_tune_type == _base.FineTuneType.SUPERVISED_FINE_TUNING.value:
        _run_supervised_fine_tuning(args=args)
