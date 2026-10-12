from __future__ import annotations

from bookworm_core.models import AskResult
from bookworm_core.openrouter import OpenRouterClient
from bookworm_core.search import SearchService


class QAService:
    def __init__(self, search: SearchService, openrouter: OpenRouterClient) -> None:
        self.search = search
        self.openrouter = openrouter

    def ask(self, question: str, *, limit: int = 8, book_filter: str | None = None) -> AskResult:
        sources = self.search.search_vector(query=question, limit=limit, book_filter=book_filter)
        answer = self.openrouter.answer_question(question, sources)
        return AskResult(answer=answer, sources=sources)
