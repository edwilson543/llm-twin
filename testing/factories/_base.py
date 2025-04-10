import factory

from llm_twin import settings
from llm_twin.domain.storage import document as document_storage
from llm_twin.domain.storage import vector as vector_storage


class Factory[T](factory.Factory):
    class Meta:
        abstract = True
        # By default, do not persist the object.
        strategy = "build"

    @classmethod
    def _create(cls, model_class: type[T], *args: object, **kwargs: object) -> T:
        """
        Persist the created object in the relevant database.
        """
        instance = model_class(*args, **kwargs)

        if isinstance(instance, document_storage.Document):
            document_db = settings.get_document_database()
            document_db.insert_one(document=instance)
        elif isinstance(instance, vector_storage.Vector):
            vector_db = settings.get_vector_database()
            vector_db.bulk_insert(vectors=[instance])
        else:
            raise NotImplementedError(f"Cannot create object of type {type(instance)}.")

        return instance
