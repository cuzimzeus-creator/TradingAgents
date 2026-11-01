"""LLM client that runs on a Claude subscription through the Claude Code CLI.

Each model call spawns ``claude -p`` (print mode) with every built-in tool
disabled, so it acts as a plain text completion backed by the user's Claude
Pro/Max plan instead of an API key. Authentication is whatever the CLI uses:
a local ``claude login``, or ``CLAUDE_CODE_OAUTH_TOKEN`` (create one with
``claude setup-token``), which is how the Docker image authenticates.

The CLI offers no native tool-calling hook for LangChain, so tool calls and
structured output are emulated: the tool/JSON-schema contract is written into
the prompt and the reply is parsed back into ``tool_calls`` / a pydantic model.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool

from .base_client import BaseLLMClient

CLAUDE_BIN_ENV = "CLAUDE_CODE_BIN"
_DEFAULT_TIMEOUT = 600

_BASE_SYSTEM = (
    "You are a text-completion engine inside an automated trading-research pipeline. "
    "You have no tools of your own. Answer exactly what the conversation asks, "
    "following any output format it specifies."
)

_TOOL_INSTRUCTIONS = """\
## Tools you may call
{tools}

To call tools, reply with ONLY one JSON object and nothing else:
{{"tool_calls": [{{"name": "<tool name>", "args": {{...}}}}]}}
You may list several calls. Arguments must follow the tool's JSON schema.
When you need no (more) tool results, reply normally with your final answer in plain text."""

_SCHEMA_INSTRUCTIONS = """\
## Output format
Reply with ONLY one JSON object (no prose, no code fence) that validates against this JSON schema:
{schema}"""


def claude_binary() -> str:
    path = os.environ.get(CLAUDE_BIN_ENV) or shutil.which("claude")
    if not path:
        raise RuntimeError(
            "The `claude` CLI was not found. Install Claude Code "
            "(npm install -g @anthropic-ai/claude-code) or set CLAUDE_CODE_BIN."
        )
    return path


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(p for p in parts if p)


def render_messages(messages: list[BaseMessage]) -> tuple[str, str]:
    """(system text, transcript) for a LangChain message list."""
    system, lines = [], []
    for m in messages:
        if isinstance(m, SystemMessage):
            system.append(_text(m.content))
        elif isinstance(m, AIMessage):
            said = _text(m.content)
            calls = "".join(
                f"\n[assistant called tool {c['name']} with {json.dumps(c['args'], default=str)}]"
                for c in (m.tool_calls or [])
            )
            lines.append(f"Assistant: {said}{calls}".rstrip())
        elif isinstance(m, ToolMessage):
            lines.append(f"[Tool result{' for ' + m.name if m.name else ''}]\n{_text(m.content)}")
        else:
            role = "User" if isinstance(m, HumanMessage) else getattr(m, "type", "user").title()
            lines.append(f"{role}: {_text(m.content)}")
    return "\n\n".join(system), "\n\n".join(lines)


def extract_json(text: str) -> Any:
    """First JSON value in ``text`` (tolerates code fences and surrounding prose)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return decoder.raw_decode(text[i:])[0]
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found in model reply")


class ChatClaudeCode(BaseChatModel):
    """Chat model backed by ``claude -p`` (a Claude subscription)."""

    model: str
    effort: str | None = None
    timeout: float = _DEFAULT_TIMEOUT
    bound_tools: list[dict] = []
    json_schema: dict | None = None

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        specs = [convert_to_openai_tool(t)["function"] for t in tools]
        return self.model_copy(update={"bound_tools": specs})

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        if hasattr(schema, "model_json_schema"):
            json_schema = schema.model_json_schema()
            parse = lambda msg: schema.model_validate(extract_json(msg.content))  # noqa: E731
        else:
            json_schema = schema.get("parameters", schema) if isinstance(schema, dict) else None
            parse = lambda msg: extract_json(msg.content)  # noqa: E731
        bound = self.model_copy(update={"json_schema": json_schema, "bound_tools": []})
        return bound | RunnableLambda(parse)

    def _prompt(self, messages: list[BaseMessage]) -> tuple[str, str]:
        system, transcript = render_messages(messages)
        extra = []
        if self.bound_tools:
            tools = "\n".join(
                f"- {t['name']}: {t.get('description', '')}\n  schema: {json.dumps(t.get('parameters', {}))}"
                for t in self.bound_tools
            )
            extra.append(_TOOL_INSTRUCTIONS.format(tools=tools))
        if self.json_schema:
            extra.append(_SCHEMA_INSTRUCTIONS.format(schema=json.dumps(self.json_schema)))
        parts = [p for p in (system, *extra) if p]
        return _BASE_SYSTEM, "\n\n".join([*parts, transcript, "Assistant:"])

    def _run_cli(self, system: str, prompt: str) -> str:
        cmd = [
            claude_binary(), "-p", "--model", self.model, "--tools", "",
            "--no-session-persistence", "--output-format", "json",
            "--system-prompt", system,
        ]
        if self.effort:
            cmd += ["--effort", self.effort]
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        # A neutral cwd keeps any project CLAUDE.md out of the model's context.
        with tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                timeout=self.timeout, cwd=cwd, env=env,
            )
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"claude CLI failed (exit {proc.returncode}): "
                               f"{(proc.stderr or proc.stdout).strip()[:500]}") from None
        if data.get("is_error") or proc.returncode != 0:
            raise RuntimeError(f"claude CLI error: {str(data.get('result') or proc.stderr)[:500]}")
        return data.get("result", "")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        system, prompt = self._prompt(messages)
        reply = self._run_cli(system, prompt).strip()
        message = AIMessage(content=reply)
        if self.bound_tools:
            message = self._as_tool_message(reply) or message
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _as_tool_message(self, reply: str) -> AIMessage | None:
        try:
            data = extract_json(reply)
        except ValueError:
            return None
        if not isinstance(data, dict) or not isinstance(data.get("tool_calls"), list):
            return None
        known = {t["name"] for t in self.bound_tools}
        calls = [
            {"name": c["name"], "args": c.get("args") or {}, "id": f"call_{uuid.uuid4().hex[:12]}",
             "type": "tool_call"}
            for c in data["tool_calls"] if isinstance(c, dict) and c.get("name") in known
        ]
        return AIMessage(content="", tool_calls=calls) if calls else None


class ClaudeCodeClient(BaseLLMClient):
    """Client for Claude models used through a Claude subscription (Claude Code CLI)."""

    def get_llm(self) -> Any:
        kwargs = {"model": self.model}
        if self.kwargs.get("effort"):
            kwargs["effort"] = self.kwargs["effort"]
        if self.kwargs.get("timeout"):
            kwargs["timeout"] = self.kwargs["timeout"]
        return ChatClaudeCode(**kwargs)

    def validate_model(self) -> bool:
        return True
