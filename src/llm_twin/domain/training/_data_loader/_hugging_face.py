import dataclasses

import datasets

from . import _base


@dataclasses.dataclass(frozen=True)
class HuggingFaceDataLoader(_base.DataLoader):
    dataset_path: str
    eos_token: str

    def load(self) -> datasets.DatasetDict:
        dataset = datasets.load_dataset(self.dataset_path, split="train")

        dataset = dataset.map(
            self._format_samples, batched=True, remove_columns=dataset.column_names
        )
        return dataset.train_test_split(test_size=0.05)

    def _format_samples(self, samples: list[dict]) -> dict[str, list[str]]:
        return {"text": [self._format_sample(sample) for sample in samples]}

    def _format_sample(self, sample: dict) -> str:
        return (
            ALPACA_TEMPLATE.format(sample["instruction"], sample["answer"])
            + self.dataset_path
        )


ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""
