import dataclasses


@dataclasses.dataclass(frozen=True)
class UserNameEmpty(Exception):
    name: str


@dataclasses.dataclass(frozen=True)
class UserName:
    first_name: str
    last_name: str


def split_user_full_name(user: str) -> UserName:
    name_tokens = user.split(" ")
    if len(name_tokens) == 0:
        raise UserNameEmpty(name=user)
    elif len(name_tokens) == 1:
        first_name, last_name = name_tokens[0], name_tokens[0]
    else:
        first_name, last_name = " ".join(name_tokens[:-1]), name_tokens[-1]

    return UserName(first_name=first_name, last_name=last_name)
