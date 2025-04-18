import pytest

from llm_twin.domain.dataset_generation import _datasets, _prompts
from testing.factories import dataset as dataset_factories
from testing.factories import vectors as vector_factories
from testing.helpers import models as models_helpers


class TestPrompt__Render:
    def test_renders_prompt_with_passed_variables(self):
        template = "A: {a}, B: {b}"
        variables = {"a": 123, "b": "xyz"}
        prompt = dataset_factories.Prompt(template=template, variables=variables)

        rendered = prompt.render()

        assert rendered == "A: 123, B: xyz"

    def test_raises_when_prompt_template_variable_is_missing(self):
        template = "A: {a}, B: {b}"
        variables = {"a": 123}
        prompt = dataset_factories.Prompt(template=template, variables=variables)

        with pytest.raises(_prompts.MissingPromptVariable) as exc:
            prompt.render()

        assert exc.value.variable_name == "b"


class TestGenerateSamplePromptFactory__CreatePromptsForGeneratingSamples:
    def test_creates_prompts_for_generating_instruct_samples(self):
        language_model = models_helpers.FakeLanguageModel()
        factory = _prompts.GenerateSamplePromptFactory(
            dataset_type=_datasets.DatasetType.INSTRUCT, language_model=language_model
        )
        repository = vector_factories.RepositoryChunk.build()
        article = vector_factories.ArticleChunk.build()
        documents = [repository, article]

        prompts = factory.create_prompts_for_generating_samples(documents=documents)

        assert len(prompts) == 2

        repository_prompt = prompts[0]
        assert repository_prompt.document == repository
        assert repository_prompt.input_data_category == repository.category()
        assert repository_prompt.variables == {"extract": repository.content}
        assert repository_prompt.response_format == _datasets.InstructSample
        assert repository_prompt.render()
        assert repository_prompt.template == _prompts.INSTRUCT_PROMPT_TEMPLATE

        article_prompt = prompts[1]
        assert article_prompt.document == article
        assert article_prompt.input_data_category == article.category()
        assert article_prompt.template == _prompts.INSTRUCT_PROMPT_TEMPLATE
        assert article_prompt.variables == {"extract": article.content}
        assert article_prompt.render()
        assert article_prompt.response_format == _datasets.InstructSample

    def test_creates_prompts_for_generating_preference_sample(self):
        language_model = models_helpers.FakeLanguageModel()
        factory = _prompts.GenerateSamplePromptFactory(
            dataset_type=_datasets.DatasetType.PREFERENCE, language_model=language_model
        )
        repository = vector_factories.RepositoryChunk.build()
        article = vector_factories.ArticleChunk.build()
        documents = [repository, article]

        prompts = factory.create_prompts_for_generating_samples(documents=documents)

        assert len(prompts) == 2

        repository_prompt = prompts[0]
        assert repository_prompt.document == repository
        assert repository_prompt.input_data_category == repository.category()
        assert repository_prompt.template == _prompts.PREFERENCE_PROMPT_TEMPLATE
        assert repository_prompt.variables == {"extract": repository.content}
        assert repository_prompt.render()
        assert repository_prompt.response_format == _datasets.PreferenceSample

        article_prompt = prompts[1]
        assert article_prompt.document == article
        assert article_prompt.input_data_category == article.category()
        assert article_prompt.template == _prompts.PREFERENCE_PROMPT_TEMPLATE
        assert article_prompt.variables == {"extract": article.content}
        assert article_prompt.render()
        assert article_prompt.response_format == _datasets.PreferenceSample


class TestGenerateSamplePromptFactory__GetSystemPrompt:
    def test_gets_system_prompt_for_generating_instruct_dataset_samples(self):
        language_model = models_helpers.FakeLanguageModel()
        factory = _prompts.GenerateSamplePromptFactory(
            dataset_type=_datasets.DatasetType.INSTRUCT, language_model=language_model
        )

        system_prompt = factory.get_system_prompt()

        assert (
            system_prompt.render()
            == "You are a helpful assistant who generates instruction-answer pairs based on the given context."
        )

    def test_gets_system_prompt_for_generating_preference_dataset_samples(self):
        language_model = models_helpers.FakeLanguageModel()
        factory = _prompts.GenerateSamplePromptFactory(
            dataset_type=_datasets.DatasetType.PREFERENCE, language_model=language_model
        )

        system_prompt = factory.get_system_prompt()

        assert (
            system_prompt.render()
            == "You are a helpful assistant who generates instruction-answer triples based on the given context."
        )
