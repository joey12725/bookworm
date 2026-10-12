from __future__ import annotations

import textwrap

import httpx

from bookworm_core.errors import OpenRouterError
from bookworm_core.models import SearchResult


class OpenRouterClient:
    def __init__(self, *, api_key: str, model: str, timeout_seconds: int = 90) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def answer_question(self, question: str, contexts: list[SearchResult]) -> str:
        if not contexts:
            raise OpenRouterError("No context passages were provided for synthesis.")

        context_lines = []
        for idx, result in enumerate(contexts, start=1):
            context_lines.append(
                textwrap.dedent(
                    f"""
                    [Source {idx}]
                    Title: {result.book_title}
                    Author: {result.author or 'Unknown'}
                    Chapter: {result.chapter_title or result.chapter_number or 'N/A'}
                    Page: {result.page_number or 'N/A'}
                    Content: {result.content}
                    """
                ).strip()
            )

        system_prompt = (
            "You answer questions using only the provided book excerpts. "
            "If evidence is insufficient, say so clearly. "
            "Cite sources inline as [Source N]."
        )

        user_prompt = (
            f"Question: {question}\n\n"
            "Context excerpts:\n"
            + "\n\n".join(context_lines)
            + "\n\nProvide a concise, accurate answer with citations."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:600]
            raise OpenRouterError(
                f"OpenRouter request failed with status {exc.response.status_code}: {body}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise OpenRouterError("OpenRouter response did not contain choices.")

        message = choices[0].get("message", {})
        content = message.get("content")
        if not content:
            raise OpenRouterError("OpenRouter response was missing message content.")

        return str(content).strip()
