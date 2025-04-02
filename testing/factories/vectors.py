import factory

from llm_twin.domain.feature_engineering import cleaning

from . import documents


class _CleanedRawDocument(factory.Factory):
    content = factory.Sequence(lambda n: f"content-{n}")
    platform = "some-platform"

    author = factory.SubFactory(documents.Author)
    author_id = factory.LazyAttribute(lambda o: str(o.author.id))
    author_full_name = factory.LazyAttribute(lambda o: o.author.full_name)


class CleanedRepository(_CleanedRawDocument):
    name = factory.Sequence(lambda n: f"repo-{n}")
    link = factory.Sequence(lambda n: f"https://github.com/edwilson543/repo-{n}/")

    class Meta:
        model = cleaning.CleanedRepository
