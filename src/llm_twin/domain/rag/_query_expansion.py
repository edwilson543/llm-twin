import pydantic

from llm_twin.domain import models


class Expansion(pydantic.BaseModel):
    query: str
    expansions: list[str]

    @property
    def all_queries(self) -> list[str]:
        return [self.query] + self.expansions


def expand_query(
    *,
    query: str,
    number_of_query_expansions: int,
    language_model: models.LanguageModel,
) -> Expansion:
    """
    Create multiple wordings of the query, to better capture nuances during context retrieval.
    """
    raise NotImplementedError


PROMPT_TEMPLATE = """Generate {number_of_query_expansions} different versions of the given query question. 
    The different versions capture nuances in the question that are useful for retrieving relevant documents 
    from a vector database. 
    
    Original query: {query}"""
