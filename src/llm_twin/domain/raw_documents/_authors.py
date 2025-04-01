from __future__ import annotations

from . import _base, _db


class Author(_base.RawDocument):
    first_name: str
    last_name: str

    @classmethod
    def get_collection_name(cls) -> _db.Collection:
        return _db.Collection.AUTHORS

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
