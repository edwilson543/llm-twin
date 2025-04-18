import dataclasses
import typing

import pydantic

from llm_twin.domain import models
from llm_twin.domain.feature_engineering import chunking
from llm_twin.domain.storage import vector as vector_storage

from . import _datasets


@dataclasses.dataclass(frozen=True)
class MissingPromptVariable(Exception):
    variable_name: str


class Prompt[ResponseFormat: pydantic.BaseModel](vector_storage.Vector):
    template: str
    variables: dict[str, typing.Any]

    class _Config(vector_storage.Config):
        category = vector_storage.DataCategory.PROMPT

    def render(self) -> str:
        try:
            return self.template.format(**self.variables)
        except KeyError as exc:
            raise MissingPromptVariable(variable_name=exc.args[0]) from exc


_SampleT = _datasets.InstructSample | _datasets.PreferenceSample


class GenerateSamplePrompt(Prompt):
    input_data_category: vector_storage.DataCategory
    document: chunking.Chunk
    response_format: type[_SampleT]


@dataclasses.dataclass(frozen=True)
class GenerateSamplePromptFactory:
    dataset_type: _datasets.DatasetType
    language_model: models.LanguageModel

    # Queries.

    def create_prompts_for_generating_samples(
        self, *, documents: list[chunking.Chunk]
    ) -> list[GenerateSamplePrompt]:
        return [self._get_prompt(document=document) for document in documents]

    def _get_system_prompt(self) -> Prompt:
        template = "You are a helpful assistant who generates {dataset_format} based on the given context."

        dataset_format = {
            _datasets.DatasetType.INSTRUCT: "instruction-answer pairs",
            _datasets.DatasetType.PREFERENCE: "instruction-answer triples",
        }[self.dataset_type]
        variables = {"dataset_format": dataset_format}

        return Prompt(template=template, variables=variables)

    def _get_prompt(self, *, document: chunking.Chunk) -> GenerateSamplePrompt:
        variables = {"extract": document.content}
        return GenerateSamplePrompt(
            template=self._get_prompt_template(),
            variables=variables,
            response_format=self._get_response_format(),
            input_data_category=document.category(),
            document=document,
        )

    def _get_prompt_template(self) -> str:
        return {
            _datasets.DatasetType.INSTRUCT: INSTRUCT_PROMPT_TEMPLATE,
            _datasets.DatasetType.PREFERENCE: PREFERENCE_PROMPT_TEMPLATE,
        }[self.dataset_type]

    def _get_response_format(self) -> type[_SampleT]:
        mapping: dict[_datasets.DatasetType, type[_SampleT]] = {
            _datasets.DatasetType.INSTRUCT: _datasets.InstructSample,
            _datasets.DatasetType.PREFERENCE: _datasets.PreferenceSample,
        }
        return mapping[self.dataset_type]



INSTRUCT_PROMPT_TEMPLATE = """Based on the following extract, generate five instruction-answer pairs. 
Each instruction must ask to write about a specific topic contained in the context. 
Each answer must provide a relevant paragraph based on the information found in the context. 
Only use concepts from the context to generate the instructions. 
Instructions must never explicitly mention a context, a system, a course, or an extract. 
Instructions must be self-contained and general. 
Answers must imitate the writing style of the context.
    
Example instruction: Explain the concept of an LLM Twin. 
Example answer: An LLM Twin is essentially an AI character that mimics your writing style, personality, and voice.
It's designed to write just like you by incorporating these elements into a language model.
The idea is to create a digital replica of your writing habits using advanced AI techniques.

Extract:
{{extract}}
"""

PREFERENCE_PROMPT_TEMPLATE = """Based on the following extract, generate five instruction-answer triples. Each triple should consist of:
1. An instruction asking about a specific topic in the context.
2. A generated answer that attempts to answer the instruction based on the context, named as 'rejected'.
3. An extracted answer that is a relevant excerpt directly from the given context, named as 'chosen'.

Instructions must be self-contained and general, without explicitly mentioning a context, system, course, or extract.

Important:
- Ensure that the extracted answer, the chosen one, is a verbatim copy from the context, including all punctuation and apostrophes.
- Do not add any ellipsis (...) or [...]  to indicate skipped text in the extracted answer.
- If the relevant text is not continuous, use two separate sentences from the context instead of skipping text.

Extract:
{{extract}}
"""
