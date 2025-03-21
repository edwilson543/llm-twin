import dataclasses
import pathlib


class CommandError(Exception):
    pass


@dataclasses.dataclass
class ConfigFileDoesNotExist(CommandError):
    filepath: pathlib.Path
