from llm_twin.domain.storage import vector as vector_storage

from . import _datasets


def filter_short_answers(
    data: dict[vector_storage.DataCategory, _datasets.PreferenceSampleDataset],
    min_length: int = 100,
) -> dict[vector_storage.DataCategory, _datasets.PreferenceSampleDataset]:
    def is_long_enough(example: _datasets.PreferenceSample) -> bool:
        return len(example.chosen) >= min_length

    filtered_data = {}
    for category, dataset in data.items():
        filetered_dataset_samples = list(filter(is_long_enough, dataset.samples))
        filtered_dataset = _datasets.PreferenceSampleDataset(
            input_data_category=category, samples=filetered_dataset_samples
        )

        filtered_data[category] = filtered_dataset

    return filtered_data


def filter_answer_format(
    data: dict[vector_storage.DataCategory, _datasets.PreferenceSampleDataset],
) -> dict[vector_storage.DataCategory, _datasets.PreferenceSampleDataset]:
    def is_valid_format(example: _datasets.PreferenceSample) -> bool:
        chosen = example.chosen

        return len(chosen) > 0 and chosen[0].isupper() and chosen[-1] in (".", "!", "?")

    filtered_data = {}
    for category, dataset in data.items():
        filetered_dataset_samples = list(filter(is_valid_format, dataset.samples))
        filtered_dataset = _datasets.PreferenceSampleDataset(
            input_data_category=category, samples=filetered_dataset_samples
        )

        filtered_data[category] = filtered_dataset

    return filtered_data
