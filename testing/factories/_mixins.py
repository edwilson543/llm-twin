import factory


class ArticleMixin(factory.Factory):
    link = factory.Sequence(lambda n: f"https://fake.com/article-{n}/")


class RepositoryMixin(factory.Factory):
    name = factory.Sequence(lambda n: f"repo-{n}")
    link = factory.Sequence(lambda n: f"https://github.com/edwilson543/repo-{n}/")
