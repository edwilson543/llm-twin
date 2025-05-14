import zenml

from llm_twin.orchestration.steps import rag as rag_steps


@zenml.pipeline
def rag_inference(query: str, max_tokens: int) -> None:
    documents = rag_steps.retrieve_documents(query=query)
    rag_steps.generate_response(query=query, documents=documents, max_tokens=max_tokens)


if __name__ == "__main__":
    rag_inference(query="What is a foreign key?", max_tokens=1000)
