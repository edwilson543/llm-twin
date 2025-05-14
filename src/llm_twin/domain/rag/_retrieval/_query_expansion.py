from __future__ import annotations

import pydantic

from llm_twin.domain import models


class Expansion(pydantic.BaseModel):
    query: str
    expansions: list[str]

    @classmethod
    def from_query(
        cls, *, query: str, n_expansions: int, language_model: models.LanguageModel
    ) -> Expansion:
        prompt_template = """Generate {n_expansions} different versions of the original query. 
            The different versions should capture nuances in the question that are useful for 
            retrieving relevant documents embedded in a vector database. 

            Original query: {query}"""

        prompt = prompt_template.format(n_expansions=n_expansions, query=query)

        return language_model.get_response(
            messages=[models.Message.user(content=prompt)],
            response_format=Expansion,
        )

    @property
    def all_queries(self) -> list[str]:
        return [self.query] + self.expansions
