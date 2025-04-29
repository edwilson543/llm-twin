import dataclasses
import pathlib

import datasets
import peft
import transformers
import trl

from llm_twin.domain import dataset_generation

from . import _base


@dataclasses.dataclass
class SupervisedFineTuning(_base.FineTuningStrategy):
    model_name: str

    # I/O.
    output_dir: pathlib.Path
    report_to: str | None = "comet_ml"

    # ML.
    dataset_text_field: str = "text"
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

    def fine_tune(self) -> None:
        dataset = self.data_loader.load_instruct_dataset()
        model, tokenizer = self._get_model_and_tokenizer()

        trainer = self._get_trainer(model=model, dataset=dataset)
        trainer.train()

        self._export_model(model=model)

    def _get_model_and_tokenizer(
        self,
    ) -> tuple[peft.PeftModel, transformers.AutoTokenizer]:
        base_model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

        peft_config = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
        )
        model = peft.get_peft_model(model=base_model, peft_config=peft_config)

        return model, tokenizer

    def _get_trainer(
        self,
        *,
        model: peft.PeftModel,
        dataset: dataset_generation.TrainTestSplit[dataset_generation.InstructSample],
    ) -> trl.SFTTrainer:
        training_args = trl.SFTConfig(
            # Data preprocessing parameters.
            dataset_text_field=self.dataset_text_field,
            dataset_num_proc=2,
            packing=True,
            # Training parameters.
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            logging_steps=1,
            optim=self.optimizer,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            per_device_eval_batch_size=self.per_device_train_batch_size,
            warmup_steps=10,
            output_dir=str(self.output_dir),
            report_to=self.report_to,
            seed=0,
        )

        train_dataset = self._format_samples(samples=dataset.train.samples)
        eval_dataset = self._format_samples(samples=dataset.test.samples)

        return trl.SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

    def _export_model(self, *, model: peft.PeftModel) -> None:
        model.save_pretrained(save_directory=str(self.output_dir))

    def _format_samples(
        self, *, samples: list[dataset_generation.InstructSample]
    ) -> datasets.Dataset:
        def _format_sample(sample: dataset_generation.InstructSample) -> str:
            return (
                ALPACA_TEMPLATE.format(sample.instruction, sample.answer)
                + self._eos_token
            )

        data = {self.dataset_text_field: [_format_sample(sample) for sample in samples]}
        return datasets.Dataset.from_dict(data)

    @property
    def _eos_token(self) -> str:
        return "ABCDEF"  # TODO TODO TODO -> EOS TOKEN.


ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""
