import dataclasses
import typing

import pydantic

from llm_twin.domain.feature_engineering import chunking
from llm_twin.domain.storage import vector as vector_storage

from . import _datasets


@dataclasses.dataclass(frozen=True)
class MissingPromptVariable(Exception):
    variable_name: str


class Prompt(vector_storage.Vector):
    template: str
    variables: dict[str, typing.Any]

    class _Config(vector_storage.Config):
        category = vector_storage.DataCategory.PROMPT

    def render(self) -> str:
        try:
            return self.template.format(**self.variables)
        except KeyError as exc:
            raise MissingPromptVariable(variable_name=exc.args[0]) from exc


class InstructSampleList(pydantic.BaseModel):
    samples: list[_datasets.InstructSample]


class PreferenceSampleList(pydantic.BaseModel):
    samples: list[_datasets.PreferenceSample]


ResponseFormatT = InstructSampleList | PreferenceSampleList


class GenerateSamplePrompt(Prompt):
    document: chunking.Chunk
    response_format: type[ResponseFormatT]

    @property
    def input_data_category(self) -> vector_storage.DataCategory:
        return self.document.category()


class GenerateSamplePromptFactory:
    """
    Dispatch a different prompt factory according to the dataset type.
    """

    def __init__(self) -> None:
        self._registry: dict[_datasets.DatasetType, _GenerateSamplePromptFactory] = {
            _datasets.DatasetType.INSTRUCT: _GenerateSamplePromptFactory(
                prompt_template=INSTRUCT_PROMPT_TEMPLATE,
                response_format=InstructSampleList,
                dataset_format="instruction-answer pairs",
            ),
            _datasets.DatasetType.PREFERENCE: _GenerateSamplePromptFactory(
                prompt_template=PREFERENCE_PROMPT_TEMPLATE,
                response_format=PreferenceSampleList,
                dataset_format="instruction-answer triples",
            ),
        }

    def create_prompts_for_generating_samples(
        self, *, dataset_type: _datasets.DatasetType, documents: list[chunking.Chunk]
    ) -> list[GenerateSamplePrompt]:
        factory = self._registry[dataset_type]
        return [factory.get_prompt(document=document) for document in documents]

        self,
        *,
        dataset_type: _datasets.DatasetType,
        documents: typing.Sequence[chunking.Chunk],
        factory = self._registry[dataset_type]
        return factory.get_system_prompt()


@dataclasses.dataclass(frozen=True)
class _GenerateSamplePromptFactory:
    """
    Create prompts for generating particular sample dataset type.
    """

    dataset_format: str
    prompt_template: str
    response_format: type[ResponseFormatT]

    def get_prompt(self, *, document: chunking.Chunk) -> GenerateSamplePrompt:
        variables = {"extract": document.content}
        return GenerateSamplePrompt(
            template=self.prompt_template,
            variables=variables,
            response_format=self.response_format,
            document=document,
        )

    def get_system_prompt(self) -> Prompt:
        template = "You are a helpful assistant who generates {dataset_format} based on the given context."
        variables = {"dataset_format": self.dataset_format}
        return Prompt(template=template, variables=variables)


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
