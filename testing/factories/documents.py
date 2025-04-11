import uuid

import factory

from llm_twin.domain import authors
from llm_twin.domain.etl import raw_documents

from . import _base, _mixins


class _Document(_base.Factory):
    pass


class Author(_Document):
    first_name = factory.Sequence(lambda n: f"{uuid.uuid4()}-{n}")
    last_name = factory.Sequence(lambda n: f"{uuid.uuid4()}-{n}")

    class Meta:
        model = authors.Author


class _ExtractedDocument(_Document):
    content = factory.LazyFunction(lambda: {"some": "content"})
    platform = "some-platform"

    author = factory.SubFactory(Author)
    author_id = factory.LazyAttribute(lambda o: o.author.id)
    author_full_name = factory.LazyAttribute(lambda o: o.author.full_name)

    class Meta:
        exclude = ("author",)


class Article(_ExtractedDocument, _mixins.ArticleMixin):
    class Meta:
        model = raw_documents.Article


class Repository(_ExtractedDocument, _mixins.RepositoryMixin):
    class Meta:
        model = raw_documents.Repository
