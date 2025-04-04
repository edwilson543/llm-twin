import factory

from llm_twin.domain import authors
from llm_twin.domain.etl import raw_documents


class _Document(factory.Factory):
    pass


class Author(_Document):
    first_name = factory.Sequence(lambda n: f"first-name-{n}")
    last_name = factory.Sequence(lambda n: f"last-name-{n}")

    class Meta:
        model = authors.Author


class _ExtractedDocument(_Document):
    content = factory.LazyFunction(dict)
    platform = "some-platform"

    author = factory.SubFactory(Author)
    author_id = factory.LazyAttribute(lambda o: o.author.id)
    author_full_name = factory.LazyAttribute(lambda o: o.author.full_name)

    class Meta:
        exclude = ("author",)


class Article(_ExtractedDocument):
    link = factory.Sequence(lambda n: f"https://fake.com/article-{n}/")

    class Meta:
        model = raw_documents.Article


class Repository(_ExtractedDocument):
    name = factory.Sequence(lambda n: f"repo-{n}")
    link = factory.Sequence(lambda n: f"https://github.com/edwilson543/repo-{n}/")

    class Meta:
        model = raw_documents.Repository
