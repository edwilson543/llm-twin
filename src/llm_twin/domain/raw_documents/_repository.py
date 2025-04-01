from . import _base, _db


class RepositoryDocument(_base.ExtractedDocument):
    name: str
    link: str

    @classmethod
    def get_collection_name(cls) -> _db.Collection:
        return _db.Collection.REPOSITORIES
