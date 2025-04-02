from . import _base, _db


class Article(_base.ExtractedDocument):
    link: str

    @classmethod
    def get_collection_name(cls) -> _db.Collection:
        return _db.Collection.ARTICLES
