import pytest

from llm_twin.domain.dataset_generation import _datasets, _prompts
from testing.factories import dataset as dataset_factories
from testing.factories import vectors as vector_factories


class TestPrompt__Render:
    def test_renders_prompt_with_passed_variables(self):
        template = "A: {a}, B: {b}"
        variables = {"a": 123, "b": "xyz"}
        prompt = dataset_factories.Prompt(template=template, variables=variables)

        rendered = prompt.render()

        assert rendered == "A: 123, B: xyz"

    @pytest.mark.parametrize(
        "template",
        [_prompts.INSTRUCT_PROMPT_TEMPLATE, _prompts.PREFERENCE_PROMPT_TEMPLATE],
    )
    def test_instruct_and_preference_prompt_templates_are_valid(self, template: str):
        extract = "Some copy definitely not in the template."
        variables = {"extract": extract}
        prompt = dataset_factories.Prompt.build(template=template, variables=variables)

        rendered = prompt.render()

        assert extract in rendered

    def test_raises_when_prompt_template_variable_is_missing(self):
        template = "A: {a}, B: {b}"
        variables = {"a": 123}
        prompt = dataset_factories.Prompt(template=template, variables=variables)

        with pytest.raises(_prompts.MissingPromptVariable) as exc:
            prompt.render()

        assert exc.value.variable_name == "b"


class TestGenerateSamplePromptFactory__CreatePromptsForGeneratingSamples:
    def test_creates_prompts_for_generating_instruct_samples(self):
        repository = vector_factories.RepositoryChunk.build()
        article = vector_factories.ArticleChunk.build()
        documents = [repository, article]

        factory = _prompts.GenerateSamplePromptFactory()

        prompts = factory.create_prompts_for_generating_samples(
            dataset_type=_datasets.DatasetType.INSTRUCT, documents=documents
        )

        assert len(prompts) == 2

        repository_prompt = prompts[0]
        assert repository_prompt.document == repository
        assert repository_prompt.input_data_category == repository.category()
        assert repository_prompt.response_format == _prompts.InstructSampleList
        assert repository_prompt.template == _prompts.INSTRUCT_PROMPT_TEMPLATE
        assert repository_prompt.variables == {"extract": repository.content}
        assert repository_prompt.render()

        article_prompt = prompts[1]
        assert article_prompt.document == article
        assert article_prompt.input_data_category == article.category()
        assert article_prompt.response_format == _prompts.InstructSampleList
        assert article_prompt.template == _prompts.INSTRUCT_PROMPT_TEMPLATE
        assert article_prompt.variables == {"extract": article.content}
        assert article_prompt.render()

    def test_creates_prompts_for_generating_preference_sample(self):
        repository = vector_factories.RepositoryChunk.build()
        article = vector_factories.ArticleChunk.build()
        documents = [repository, article]

        factory = _prompts.GenerateSamplePromptFactory()

        prompts = factory.create_prompts_for_generating_samples(
            dataset_type=_datasets.DatasetType.PREFERENCE, documents=documents
        )

        assert len(prompts) == 2

        repository_prompt = prompts[0]
        assert repository_prompt.document == repository
        assert repository_prompt.input_data_category == repository.category()
        assert repository_prompt.response_format == _prompts.PreferenceSampleList
        assert repository_prompt.template == _prompts.PREFERENCE_PROMPT_TEMPLATE
        assert repository_prompt.variables == {"extract": repository.content}
        assert repository_prompt.render()

        article_prompt = prompts[1]
        assert article_prompt.document == article
        assert article_prompt.input_data_category == article.category()
        assert article_prompt.response_format == _prompts.PreferenceSampleList
        assert article_prompt.template == _prompts.PREFERENCE_PROMPT_TEMPLATE
        assert article_prompt.variables == {"extract": article.content}
        assert article_prompt.render()


class TestGenerateSamplePromptFactory__GetSystemPrompt:
    def test_gets_system_prompt_for_generating_instruct_dataset_samples(self):
        factory = _prompts.GenerateSamplePromptFactory()

        system_prompt = factory.get_system_prompt(
            dataset_type=_datasets.DatasetType.INSTRUCT
        )

        assert (
            system_prompt.render()
            == "You are a helpful assistant who generates instruction-answer pairs based on the given context."
        )

    def test_gets_system_prompt_for_generating_preference_dataset_samples(self):
        factory = _prompts.GenerateSamplePromptFactory()

        system_prompt = factory.get_system_prompt(
            dataset_type=_datasets.DatasetType.PREFERENCE
        )

        assert (
            system_prompt.render()
            == "You are a helpful assistant who generates instruction-answer triples based on the given context."
        )
