from . import _base, _db


class Repository(_base.ExtractedDocument):
    name: str
    link: str

    @classmethod
    def get_collection_name(cls) -> _db.Collection:
        return _db.Collection.REPOSITORIES
