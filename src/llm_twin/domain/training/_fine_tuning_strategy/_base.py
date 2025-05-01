import abc
import dataclasses
import pathlib

from llm_twin.domain import dataset_generation


@dataclasses.dataclass
class FineTuningStrategy[_SampleT: dataset_generation.SampleT](abc.ABC):
    model_name: str

    # I/O.
    report_to: str | None
    output_dir: pathlib.Path

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
    def fine_tune(
        self, *, dataset: dataset_generation.TrainTestSplit[_SampleT]
    ) -> None:
        raise NotImplementedError
