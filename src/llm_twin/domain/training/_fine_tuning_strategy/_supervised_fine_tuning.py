import dataclasses
import pathlib

import transformers
import trl

from . import _base


class UnslothNotSupportOnMacOs(ImportError):
    pass


try:
    import unsloth
    from unsloth import chat_templates as unsloth_chat_templates
except ImportError as exc:
    raise UnslothNotSupportOnMacOs from exc


@dataclasses.dataclass
class SupervisedFineTuning(_base.FineTuningStrategy):
    # I/O.
    output_dir: pathlib.Path
    dataset_huggingface_workspace: str
    export_repo_id: str
    model_name: str

    # ML.
    learning_rate: float = 3e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 2048
    load_in_4bit: bool = False
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    chat_template: str = "chatml"
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
        dataset = self.data_loader.load()

        self._load_model()

        trainer = trl.SFTTrainer(
            model=self._model,
            tokenizer=self._tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=True,
            args=self._training_args(),
        )
        trainer.train()

        self._export_model()

    def _load_model(self) -> None:
        model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
        )

        self._model = unsloth.FastLanguageModel.get_peft_model(
            model,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
        )

        self._tokenizer = unsloth_chat_templates.get_chat_template(
            tokenizer, chat_template=self.chat_template
        )

    def _export_model(self):
        self._model.save_pretrained_merged(
            self.output_dir, self._tokenizer, save_method="merged_16bit"
        )
        self._model.push_to_hub_merged(
            self.export_repo_id, self._tokenizer, save_method="merged_16bit"
        )

    def _training_args(self) -> transformers.TrainingArguments:
        return transformers.TrainingArguments(
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=not unsloth.is_bfloat16_supported(),
            bf16=unsloth.is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            per_device_eval_batch_size=self.per_device_train_batch_size,
            warmup_steps=10,
            output_dir=self.output_dir,
            report_to="comet_ml",
            seed=0,
        )
