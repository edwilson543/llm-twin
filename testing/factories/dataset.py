import factory

from llm_twin.domain import dataset_generation

from . import _base, vectors


class Prompt(_base.Factory):
    template = factory.Sequence(lambda n: f"template-{n}")
    variables = factory.LazyFunction(dict)

    class Meta:
        model = dataset_generation.Prompt


class _GenerateSamplePrompt(Prompt):
    document = factory.SubFactory(vectors.ArticleChunk)

    class Meta:
        abstract = True
        model = dataset_generation.GenerateSamplePrompt


class GenerateInstructSamplePrompt(_GenerateSamplePrompt):
    dataset_type = dataset_generation.DatasetType.INSTRUCT


class GeneratePreferenceSamplePrompt(_GenerateSamplePrompt):
    dataset_type = dataset_generation.DatasetType.PREFERENCE


class InstructSample(_base.Factory):
    instruction = factory.Sequence(lambda n: f"instruction-{n}")
    answer = factory.Sequence(lambda n: f"answer-{n}")

    class Meta:
        model = dataset_generation.InstructSample


SAMPLES_PER_PROMPT = 5


class InstructSampleList(_base.Factory):
    samples = factory.LazyFunction(
        lambda: [InstructSample() for _ in range(0, SAMPLES_PER_PROMPT)]
    )

    class Meta:
        model = dataset_generation.InstructSampleList


class PreferenceSampleList(_base.Factory):
    samples = factory.LazyFunction(
        lambda: [PreferenceSample() for _ in range(0, SAMPLES_PER_PROMPT)]
    )

    class Meta:
        model = dataset_generation.PreferenceSampleList


class PreferenceSample(_base.Factory):
    instruction = factory.Sequence(lambda n: f"instruction-{n}")
    rejected = factory.Sequence(lambda n: f"rejected-{n}")
    chosen = factory.Sequence(lambda n: f"chosen-{n}")

    class Meta:
        model = dataset_generation.PreferenceSample


# Datasets.


class _SampleDataset(_base.Factory):
    class Meta:
        model = dataset_generation.SampleDataset


class InstructSampleDataset(_SampleDataset):
    dataset_type = dataset_generation.DatasetType.INSTRUCT
    samples = factory.List([InstructSample.build() for _ in range(2)])


class PreferenceSampleDataset(_SampleDataset):
    dataset_type = dataset_generation.DatasetType.PREFERENCE
    samples = factory.List([PreferenceSample.build() for _ in range(2)])


# Split datasets.


class _TrainTestSplit(_base.Factory):
    class Meta:
        model = dataset_generation.TrainTestSplit


class InstructTrainTestSplit(_TrainTestSplit):
    dataset_type = dataset_generation.DatasetType.INSTRUCT
    train = factory.SubFactory(InstructSampleDataset)
    test = factory.SubFactory(InstructSampleDataset)


class PreferenceTrainTestSplit(_TrainTestSplit):
    dataset_type = dataset_generation.DatasetType.PREFERENCE
    train = factory.SubFactory(PreferenceSampleDataset)
    test = factory.SubFactory(PreferenceSampleDataset)
