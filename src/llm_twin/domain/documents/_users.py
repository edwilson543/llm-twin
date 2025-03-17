from __future__ import annotations

from . import _base


class UserDocument(_base.NoSQLDocument):
    first_name: str
    last_name: str

    @classmethod
    def _get_collection_name(cls) -> str:
        return "users"

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
