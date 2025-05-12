from . import _config, _retrieval


def augment_query(*, query: str, config: _config.RAGConfig) -> str:
    """
    Augment a query with context retrieved from the database.
    """
    chunks = _retrieval.retrieve_context_for_query(query=query, config=config)
    context = "\n".join(chunk.to_context() for chunk in chunks)
    return PROMPT_TEMPLATE.format(query=query, context=context)


PROMPT_TEMPLATE = """
Answer the query below using the provided context as the primary source of information.

Query: {query}
Context: {context}
"""
