import dataclasses

from llm_twin.domain import documents


@dataclasses.dataclass(frozen=True)
class UnableToCrawlLink(Exception):
    link: str


class Crawler:
    def extract(self, *, link: str, user: documents.UserDocument) -> None:
        pass
