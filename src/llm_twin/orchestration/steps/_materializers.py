import functools
import json
import os

import pydantic
from zenml import materializers
from zenml.materializers import materializer_registry
from zenml.utils import yaml_utils

from llm_twin.domain.etl import raw_documents
from llm_twin.domain.feature_engineering import chunking, cleaning, embedding


DEFAULT_FILENAME = "data.json"

# TODO -> fix from the clean raw documents step.


class PolymorphicPydanticMaterializer(materializers.PydanticMaterializer):
    def save(self, data: pydantic.BaseModel) -> None:
        model_dump = data.model_dump()
        model_dump["_type"] = type(data).__name__
        serialized = json.dumps(model_dump)

        yaml_utils.write_json(self._data_path, serialized)

    def load(self, data_type: type[pydantic.BaseModel]):
        serialized = yaml_utils.read_json(self._data_path)
        contents = json.loads(serialized)

        _type = contents.pop("_type")
        data_type = self._data_type_lookup.get(_type, data_type)

        return data_type.model_validate(contents)

    @property
    def _data_path(self) -> str:
        return os.path.join(self.uri, DEFAULT_FILENAME)

    @functools.cached_property
    def _data_type_lookup(self) -> dict[str, type[pydantic.BaseModel]]:
        types: list[type[pydantic.BaseModel]] = [
            raw_documents.RawDocument,
            raw_documents.Article,
            raw_documents.Repository,
            cleaning.CleanedDocument,
            cleaning.CleanedArticle,
            cleaning.CleanedRepository,
            chunking.Chunk,
            chunking.ArticleChunk,
            chunking.RepositoryChunk,
            embedding.EmbeddedChunk,
            embedding.EmbeddedArticleChunk,
            embedding.EmbeddedRepositoryChunk,
        ]

        return {data_type.__name__: data_type for data_type in types}


materializer_registry.materializer_registry.register_and_overwrite_type(
    key=pydantic.BaseModel, type_=PolymorphicPydanticMaterializer
)
materializer_registry.materializer_registry.register_and_overwrite_type(
    key=raw_documents.RawDocument, type_=PolymorphicPydanticMaterializer
)
