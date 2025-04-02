from __future__ import annotations

import abc

import pydantic

from llm_twin.domain.storage import document


class RawDocument(document.Document, abc.ABC):
    """
    A raw document that was extracted from some webpage by a crawler.
    """

    content: dict
    platform: str
    author_id: pydantic.UUID4
    author_full_name: str
