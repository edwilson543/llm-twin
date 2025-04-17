import pytest

from llm_twin.domain.dataset_generation import _prompts
from testing.factories import dataset as dataset_factories


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
