import typing

import fastapi
import pydantic
from fastapi import applications

from llm_twin import config
from llm_twin.domain import rag


app = applications.FastAPI()


class Request(pydantic.BaseModel):
    query: str
    max_tokens: int = 4096
    top_k: int | None = None


class Response(pydantic.BaseModel):
    completion: str


@app.post("/completions")
async def completion(
    request: typing.Annotated[Request, fastapi.Body(embed=False)],
) -> Response:
    retrieval_config = config.get_retrieval_config()
    model = config.get_inference_engine()

    documents = rag.retrieve_relevant_documents(
        query=request.query, config=retrieval_config
    )
    instruction = rag.augment_query(query=request.query, documents=documents)

    _, completion = model.get_response(
        instruction=instruction, max_tokens=request.max_tokens
    )

    return Response(completion=completion)
