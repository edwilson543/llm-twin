import typing

from llm_twin.domain.feature_engineering import embedding


def augment_query(
    *, query: str, documents: typing.Sequence[embedding.EmbeddedChunk]
) -> str:
    """
    Augment a query with context retrieved from the database.
    """
    context = "\n".join(document.to_context() for document in documents)
    return PROMPT_TEMPLATE.format(query=query, context=context)


PROMPT_TEMPLATE = """
Answer the query below using the provided context as the primary source of information.

Query: {query}
Context: {context}
"""
