from __future__ import annotations

import json

import httpx
import pytest

from bookworm import Bookworm
from bookworm.config import Settings
from bookworm.errors import BookwormError
from bookworm.llm import Message, OpenAICompatibleChat, create_chat_model
from bookworm.qa import cited_numbers, extractive_answer, format_sources


class FakeChat:
    name = "fake"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list[Message] = []

    def complete(self, messages: list[Message]) -> str:
        self.messages = messages
        return self.reply


def test_cited_numbers() -> None:
    assert cited_numbers("A [2]. B [1, 3]. C [2] [9]", 3) == [2, 1, 3]


def test_extractive_answer_cites_sources(bw: Bookworm) -> None:
    result = bw.ask("What did Alice drink from the bottle labelled DRINK ME?")
    assert result.mode == "extractive"
    assert result.cited
    assert "[1]" in result.text or "[2]" in result.text
    assert result.sources[0].citation.book == "Alice's Adventures in Wonderland"


def test_extractive_answer_with_no_overlap() -> None:
    assert "couldn't find" in extractive_answer("zebra", [])


def test_llm_answer_gets_numbered_sources(bw: Bookworm) -> None:
    chat = FakeChat("Sun Tzu ranks it highest to win without fighting [1].")
    bw._chat, bw._chat_resolved = chat, True
    result = bw.ask("What is the supreme excellence in war?", limit=3)
    assert result.mode == "llm"
    assert result.cited == [1]
    system, user = chat.messages
    assert "only the numbered source passages" in system["content"]
    assert user["content"].startswith("Sources:\n\n[1] The Art of War")
    assert "Question: What is the supreme excellence in war?" in user["content"]


def test_format_sources(bw: Bookworm) -> None:
    hits = bw.search("rabbit", limit=2)
    text = format_sources(hits)
    assert text.startswith("[1] Alice's Adventures in Wonderland")
    assert "\n\n[2] " in text


def test_openai_compatible_chat() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": " Hi [1] "}}]})

    chat = OpenAICompatibleChat(
        base_url="https://llm.test/v1",
        model="tiny",
        api_key="k",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert chat.complete([{"role": "user", "content": "q"}]) == "Hi [1]"
    assert seen["url"] == "https://llm.test/v1/chat/completions"
    assert seen["model"] == "tiny"


def test_create_chat_model() -> None:
    assert create_chat_model(Settings()) is None
    chat = create_chat_model(Settings(chat_provider="ollama"))
    assert chat is not None and chat.name == "llama3.2"
    with pytest.raises(BookwormError, match="OPENROUTER_API_KEY"):
        create_chat_model(Settings(chat_provider="openrouter"))
