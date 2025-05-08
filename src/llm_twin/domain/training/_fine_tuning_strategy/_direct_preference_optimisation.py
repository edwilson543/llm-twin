import dataclasses

import datasets
import peft
import transformers
import trl

from llm_twin.domain import dataset_generation

from . import _base


@dataclasses.dataclass
class DirectPreferenceOptimisation(
    _base.FineTuningStrategy[dataset_generation.PreferenceSample]
):
    beta: float = 0.5

    def fine_tune(
        self,
        *,
        dataset: dataset_generation.TrainTestSplit[dataset_generation.PreferenceSample],
    ) -> None:
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

        train_dataset = self._format_samples(
            samples=dataset.train.samples, eos_token=tokenizer.eos_token
        )
        eval_dataset = self._format_samples(
            # TODO -> split out a separate validation set here instead of using test set.
            samples=dataset.test.samples, eos_token=tokenizer.eos_token
        )

        return trl.DPOTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

    def _export_model(self, *, model: peft.PeftModel) -> None:
        model.merge_and_unload()
        model.save_pretrained(save_directory=str(self.output_dir))

    @staticmethod
    def _format_samples(
        *, samples: list[dataset_generation.PreferenceSample], eos_token: str
    ) -> datasets.Dataset:
        def _format_sample(
            sample: dataset_generation.PreferenceSample,
        ) -> dict[str, str]:
            return {
                "prompt": ALPACA_TEMPLATE.format(sample.instruction, ""),
                "chosen": sample.chosen + eos_token,
                "rejected": sample.rejected + eos_token,
            }

        formatted_data = [_format_sample(sample) for sample in samples]
        return datasets.Dataset.from_list(formatted_data)


ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""
