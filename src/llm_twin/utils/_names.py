from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class NameIsEmpty(Exception):
    name: str


@dataclasses.dataclass(frozen=True)
class Name:
    first_name: str
    last_name: str

    @classmethod
    def from_full_name(cls, full_name: str) -> Name:
        name_tokens = full_name.split(" ")
        if len(name_tokens) == 0:
            raise NameIsEmpty(name=full_name)
        elif len(name_tokens) == 1:
            first_name, last_name = name_tokens[0], name_tokens[0]
        else:
            first_name, last_name = " ".join(name_tokens[:-1]), name_tokens[-1]

        return Name(first_name=first_name, last_name=last_name)
