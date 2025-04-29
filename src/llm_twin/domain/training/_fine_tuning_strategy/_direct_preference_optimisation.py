import dataclasses
import pathlib

import datasets
import peft
import transformers
import trl

from llm_twin.domain import dataset_generation

from . import _base


@dataclasses.dataclass
class DirectPreferenceOptimisation(_base.FineTuningStrategy):
    model_name: str

    # I/O.
    output_dir: pathlib.Path
    report_to: str | None = "comet_ml"

    # ML.

    beta: float = 0.5
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
        dataset = self.data_loader.load_preference_dataset()
        model, tokenizer = self._get_model_and_tokenizer()

        trainer = self._get_trainer(model=model, tokenizer=tokenizer, dataset=dataset)
        trainer.train()

        self._export_model(model=model)

    def _get_model_and_tokenizer(
        self,
    ) -> tuple[peft.PeftModel, transformers.AutoTokenizer]:
        base_model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name)

        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token

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
        tokenizer: transformers.AutoTokenizer,
        dataset: dataset_generation.TrainTestSplit[dataset_generation.PreferenceSample],
    ) -> trl.DPOTrainer:
        training_args = trl.DPOConfig(
            # Data preprocessing parameters.
            dataset_num_proc=2,
            # Training parameters.
            beta=self.beta,
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

        return trl.DPOTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

    def _export_model(self, *, model: peft.PeftModel) -> None:
        model.save_pretrained(save_directory=str(self.output_dir))

    def _format_samples(
        self, *, samples: list[dataset_generation.PreferenceSample]
    ) -> datasets.Dataset:
        def _format_sample(
            sample: dataset_generation.PreferenceSample,
        ) -> dict[str, str]:
            return {
                "prompt": ALPACA_TEMPLATE.format(sample.instruction, ""),
                "chosen": sample.chosen + self._eos_token,
                "rejected": sample.rejected + self._eos_token,
            }

        formatted_data = [_format_sample(sample) for sample in samples]
        return datasets.Dataset.from_list(formatted_data)

    @property
    def _eos_token(self) -> str:
        return "ABCDEF"  # TODO TODO TODO -> EOS TOKEN.


ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""
