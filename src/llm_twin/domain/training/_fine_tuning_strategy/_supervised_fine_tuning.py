import dataclasses

import datasets
import peft
import transformers
import trl

from llm_twin.domain import dataset_generation

from . import _base


@dataclasses.dataclass
class SupervisedFineTuning(_base.FineTuningStrategy[dataset_generation.InstructSample]):
    dataset_text_field: str = "text"

    def fine_tune(
        self,
        *,
        dataset: dataset_generation.TrainTestSplit[dataset_generation.InstructSample],
    ) -> None:
        model, tokenizer = self._get_model_and_tokenizer()

        trainer = self._get_trainer(model=model, tokenizer=tokenizer, dataset=dataset)
        trainer.train()

        model.merge_and_unload()
        model.save_pretrained(save_directory=str(self.output_dir))
        tokenizer.save_pretrained(save_directory=str(self.output_dir))

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
        tokenizer: transformers.AutoTokenizer,
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

        train_dataset = self._format_samples(
            samples=dataset.train.samples, eos_token=tokenizer.eos_token
        )
        eval_dataset = self._format_samples(
            samples=dataset.test.samples, eos_token=tokenizer.eos_token
        )

        return trl.SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

    def _format_samples(
        self, *, samples: list[dataset_generation.InstructSample], eos_token: str
    ) -> datasets.Dataset:
        def _format_sample(sample: dataset_generation.InstructSample) -> str:
            return ALPACA_TEMPLATE.format(sample.instruction, sample.answer) + eos_token

        data = {self.dataset_text_field: [_format_sample(sample) for sample in samples]}
        return datasets.Dataset.from_dict(data)


ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""
