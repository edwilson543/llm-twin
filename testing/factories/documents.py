import uuid

import factory

from llm_twin.domain import raw_documents


class _Document(factory.Factory):
    id = factory.LazyFunction(uuid.uuid4)


class UserDocument(_Document):
    first_name = factory.Sequence(lambda n: f"first-name-{n}")
    last_name = factory.Sequence(lambda n: f"last-name-{n}")

    class Meta:
        model = raw_documents.UserDocument
