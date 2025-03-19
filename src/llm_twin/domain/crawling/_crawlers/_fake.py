from llm_twin.domain import documents

from . import _base


class FakeCrawler(_base.Crawler):
    def extract(self, *, link: str, user: documents.UserDocument) -> None:
        return None
