import factory

from llm_twin.domain import dataset_generation
from llm_twin.domain.storage import vector as vector_storage


class Prompt(factory.Factory):
    template = factory.Sequence(lambda n: f"template-{n}")
    variables = factory.LazyFunction(dict)

    class Meta:
        model = dataset_generation.Prompt


class InstructSample(factory.Factory):
    instruction = factory.Sequence(lambda n: f"instruction-{n}/")
    answer = factory.Sequence(lambda n: f"answer-{n}")

    class Meta:
        model = dataset_generation.InstructSample


class InstructDataset(factory.Factory):
    input_data_category = vector_storage.DataCategory.TESTING
    samples = factory.LazyFunction(list)

    class Meta:
        model = dataset_generation.InstructSampleDataset
