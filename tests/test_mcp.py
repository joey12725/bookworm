from __future__ import annotations

import io
import json

from bookworm import Bookworm
from bookworm.mcp import MCPServer, serve


def call(server: MCPServer, name: str, **arguments: object) -> dict:
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    assert response is not None and response["id"] == 7
    return response["result"]


def test_initialize_and_list_tools(bw: Bookworm) -> None:
    server = MCPServer(bw)
    init = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-03-26"},
        }
    )
    assert init is not None
    assert init["result"]["protocolVersion"] == "2025-03-26"
    assert init["result"]["serverInfo"]["name"] == "bookworm"
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    tools = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert tools is not None
    names = {t["name"] for t in tools["result"]["tools"]}
    assert names == {"search_library", "ask_library", "list_books", "read_chapter"}


def test_search_tool(bw: Bookworm) -> None:
    result = call(MCPServer(bw), "search_library", query="Curiouser and curiouser", limit=2)
    assert result["isError"] is False
    assert result["content"][0]["text"].startswith("[1] Alice's Adventures in Wonderland")
    assert len(result["structuredContent"]["results"]) == 2


def test_read_and_list_tools(bw: Bookworm) -> None:
    server = MCPServer(bw)
    books = call(server, "list_books")
    assert "The Art of War by Sun Tzu" in books["content"][0]["text"]
    chapter = call(server, "read_chapter", book="art of war", chapter=2)
    assert chapter["structuredContent"]["chapter_title"] == "Chapter II. Waging War"


def test_ask_tool(bw: Bookworm) -> None:
    result = call(MCPServer(bw), "ask_library", question="Who is Mr. Bingley?")
    assert "Sources:" in result["content"][0]["text"]
    assert result["structuredContent"]["mode"] == "extractive"


def test_tool_errors_are_in_band(bw: Bookworm) -> None:
    server = MCPServer(bw)
    result = call(server, "read_chapter", book="Nonexistent Book", chapter=1)
    assert result["isError"] is True
    missing_arg = call(server, "search_library")
    assert missing_arg["isError"] is True


def test_unknown_method_and_tool(bw: Bookworm) -> None:
    server = MCPServer(bw)
    response = server.handle({"jsonrpc": "2.0", "id": 3, "method": "resources/list"})
    assert response is not None and response["error"]["code"] == -32601
    response = server.handle(
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "nope"}}
    )
    assert response is not None and response["error"]["code"] == -32602


def test_stdio_loop(bw: Bookworm) -> None:
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "ping"},
    ]
    stdin = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\nnot json\n")
    stdout = io.StringIO()
    serve(bw, stdin=stdin, stdout=stdout)
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [line.get("id") for line in lines] == [1, 2, None]
    assert lines[1]["result"] == {}
    assert lines[2]["error"]["code"] == -32700


def test_mcp_command_serves_read_only(bw: Bookworm, monkeypatch) -> None:
    from click.testing import CliRunner

    from bookworm.cli import cli

    db = bw.store.path
    bw.close()
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "list_books", "arguments": {}}}  # fmt: skip
    monkeypatch.setenv("BOOKWORM_HASH_DIM", "256")
    result = CliRunner().invoke(cli, ["--db", str(db), "mcp"], input=json.dumps(request) + "\n")
    assert result.exit_code == 0, result.output
    assert "Pride and Prejudice" in json.loads(result.output)["result"]["content"][0]["text"]
    missing = CliRunner().invoke(cli, ["--db", str(db.parent / "nope.duckdb"), "mcp"])
    assert missing.exit_code == 1 and "No library" in missing.output
