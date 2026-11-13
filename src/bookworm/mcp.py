"""A Model Context Protocol server over stdio, so agents (Claude Desktop, Claude Code,
Cursor, ...) can search and read your library.

Implemented directly on JSON-RPC 2.0 with no SDK dependency. Register it with your client
as the command ``bookworm mcp`` (set ``BOOKWORM_DB`` to point at your library).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import IO, Any

from bookworm import __version__
from bookworm.app import Bookworm
from bookworm.errors import BookwormError

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_library",
        "description": (
            "Search the user's ebook library and return the most relevant passages with "
            "citations (book, chapter, page). Use double quotes for exact phrases."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                "book": {"type": "string", "description": "Optional book title filter."},
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "keyword", "vector"],
                    "default": "hybrid",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "ask_library",
        "description": (
            "Answer a question from the user's books. Returns an answer with numbered "
            "citations plus the source passages."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "book": {"type": "string", "description": "Optional book title filter."},
            },
            "required": ["question"],
        },
    },
    {
        "name": "list_books",
        "description": "List every book in the library with author and chapter count.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_chapter",
        "description": "Return the full text of one chapter. Use list_books or "
        "search_library first to find the book and chapter number.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "book": {"type": "string"},
                "chapter": {"type": "integer", "minimum": 1},
            },
            "required": ["book", "chapter"],
        },
    },
]


class MCPServer:
    def __init__(self, bookworm: Bookworm) -> None:
        self.bw = bookworm
        self._tools: dict[str, Callable[[dict[str, Any]], tuple[str, Any]]] = {
            "search_library": self._search,
            "ask_library": self._ask,
            "list_books": self._list_books,
            "read_chapter": self._read_chapter,
        }

    # -- tools ------------------------------------------------------------------------

    def _search(self, args: dict[str, Any]) -> tuple[str, Any]:
        hits = self.bw.search(
            str(args["query"]),
            limit=int(args.get("limit", 5)),
            mode=args.get("mode", "hybrid"),
            book=args.get("book"),
        )
        if not hits:
            return "No matching passages.", {"results": []}
        text = "\n\n---\n\n".join(
            f"[{n}] {h.citation.format()}\n\n{h.text}" for n, h in enumerate(hits, start=1)
        )
        return text, {"results": [h.to_dict() for h in hits]}

    def _ask(self, args: dict[str, Any]) -> tuple[str, Any]:
        result = self.bw.ask(str(args["question"]), book=args.get("book"))
        sources = "\n".join(
            f"[{n}] {h.citation.format()}" for n, h in enumerate(result.sources, start=1)
        )
        return f"{result.text}\n\nSources:\n{sources}", result.to_dict()

    def _list_books(self, args: dict[str, Any]) -> tuple[str, Any]:
        books = self.bw.books()
        lines = [
            f"- {b.title}" + (f" by {b.author}" if b.author else "") + f" ({b.chapters} chapters)"
            for b in books
        ]
        data = [
            {"title": b.title, "author": b.author, "chapters": b.chapters, "format": b.format}
            for b in books
        ]
        return "\n".join(lines) or "The library is empty.", {"books": data}

    def _read_chapter(self, args: dict[str, Any]) -> tuple[str, Any]:
        book, title, text = self.bw.read(str(args["book"]), int(args["chapter"]))
        heading = f"{book.title}: {title or 'Chapter ' + str(args['chapter'])}"
        return f"{heading}\n\n{text}", {"book": book.title, "chapter_title": title, "text": text}

    # -- protocol ---------------------------------------------------------------------

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Process one JSON-RPC message; returns the response, or None for notifications."""
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}
        if msg_id is None:  # notification (e.g. notifications/initialized)
            return None
        try:
            result = self._dispatch(str(method), params)
        except _RPCError as exc:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": exc.code, "message": str(exc)},
            }
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            requested = params.get("protocolVersion")
            version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
            return {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "bookworm", "version": __version__},
                "instructions": "Search and read the user's personal ebook library. "
                "Cite passages using the citation labels returned by the tools.",
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": TOOLS}
        if method == "tools/call":
            name = params.get("name")
            tool = self._tools.get(str(name))
            if tool is None:
                raise _RPCError(-32602, f"Unknown tool: {name}")
            try:
                text, structured = tool(params.get("arguments") or {})
            except (BookwormError, KeyError, ValueError, TypeError) as exc:
                # Tool errors are reported in-band so the model can recover.
                return {"content": [{"type": "text", "text": f"Error: {exc}"}], "isError": True}
            return {
                "content": [{"type": "text", "text": text}],
                "structuredContent": structured
                if isinstance(structured, dict)
                else {"data": structured},
                "isError": False,
            }
        raise _RPCError(-32601, f"Method not found: {method}")


class _RPCError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def serve(bookworm: Bookworm, stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> None:
    """Run the newline-delimited JSON-RPC loop until stdin closes."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    server = MCPServer(bookworm)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response: dict[str, Any] | None = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
        else:
            response = server.handle(message) if isinstance(message, dict) else None
        if response is not None:
            stdout.write(json.dumps(response, default=str) + "\n")
            stdout.flush()
