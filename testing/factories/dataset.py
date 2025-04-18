import factory

from llm_twin.domain import dataset_generation
from llm_twin.domain.storage import vector as vector_storage

from . import _base, vectors


class Prompt(_base.Factory):
    template = factory.Sequence(lambda n: f"template-{n}")
    variables = factory.LazyFunction(dict)

    class Meta:
        model = dataset_generation.Prompt


class _GenerateInstructSamplePrompt(Prompt):
    input_data_category = vector_storage.DataCategory.ARTICLES
    document = factory.SubFactory(vectors.ArticleChunk)

    class Meta:
        abstract = True
        model = dataset_generation.GenerateSamplePrompt


class GenerateInstructSamplePrompt(_GenerateInstructSamplePrompt):
    response_format = dataset_generation.InstructSample


class GeneratePreferenceSamplePrompt(_GenerateInstructSamplePrompt):
    response_format = dataset_generation.PreferenceSample


class InstructSample(_base.Factory):
    instruction = factory.Sequence(lambda n: f"instruction-{n}")
    answer = factory.Sequence(lambda n: f"answer-{n}")

    class Meta:
        model = dataset_generation.InstructSample


class PreferenceSample(_base.Factory):
    instruction = factory.Sequence(lambda n: f"instruction-{n}")
    rejected = factory.Sequence(lambda n: f"rejected-{n}")
    chosen = factory.Sequence(lambda n: f"chosen-{n}")

    class Meta:
        model = dataset_generation.PreferenceSample


class SampleDataset(_base.Factory):
    input_data_category = vector_storage.DataCategory.TESTING
    samples = factory.LazyFunction(list)

    class Meta:
        model = dataset_generation.SampleDataset
