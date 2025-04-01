from . import _base, _db


class ArticleDocument(_base.ExtractedDocument):
    link: str

    @classmethod
    def _get_collection_name(cls) -> _db.Collection:
        return _db.Collection.ARTICLES
