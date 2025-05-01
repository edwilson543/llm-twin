import abc
import dataclasses
import pathlib

from llm_twin.domain.training import _data_loader


@dataclasses.dataclass
class FineTuningStrategy(abc.ABC):
    model_name: str

    # I/O.
    data_loader: _data_loader.DataLoader
    output_dir: pathlib.Path
    report_to: str | None = "comet_ml"

    # ML.
    learning_rate: float = 3e-4
    num_train_epochs: int = 3
    optimizer: str = "adamw_8bit"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] = dataclasses.field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "up_proj",
            "down_proj",
            "o_proj",
            "gate_proj",
        ]
    )

    @abc.abstractmethod
    def fine_tune(self) -> None:
        raise NotImplementedError
