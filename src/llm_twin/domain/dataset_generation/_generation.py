from __future__ import annotations

from llm_twin.domain import models

from . import _datasets, _prompts


def generate_sample_dataset(
    *,
    dataset_type: _datasets.DatasetType,
    language_model: models.LanguageModel,
    system_prompt: _prompts.Prompt,
    prompts: list[_prompts.GenerateSamplePrompt],
) -> _datasets.SampleDataset:
    system_message = models.Message.system(content=system_prompt.render())

    samples: list[_datasets.SampleT] = []

    for prompt in prompts:
        user_message = models.Message.user(content=prompt.render())

        response = language_model.get_response(
            messages=[system_message, user_message],
            response_format=prompt.response_format,
        )
        samples.extend(response.samples)

    return _datasets.SampleDataset(samples=samples, dataset_type=dataset_type)
