import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Union

from dotenv import load_dotenv

load_dotenv(override=True)

PROVIDER = os.getenv("PROVIDER", "ollama")
MODEL = os.getenv("MODEL", os.getenv("OLLAMA_MODEL", "qwen3"))
FAST_MODEL = os.getenv("FAST_MODEL", os.getenv("OLLAMA_FAST_MODEL", ""))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# How long Ollama keeps the model loaded after each request.
# 0 = unload immediately (default), -1 = forever, "30m" = duration, "server" = use the server's OLLAMA_KEEP_ALIVE
_keep_alive = os.getenv("KEEP_ALIVE", "0").strip()
KEEP_ALIVE = None if _keep_alive in ("", "server") else int(_keep_alive) if _keep_alive.lstrip("-").isdigit() else _keep_alive


# ── Normalized response types (mirror Ollama's response.message shape) ────────

@dataclass
class NormalizedToolFunction:
    name: str
    arguments: dict  # always a dict; OpenAI returns JSON string — we convert it


@dataclass
class NormalizedToolCall:
    function: NormalizedToolFunction
    tool_use_id: Optional[str] = None  # Anthropic only


@dataclass
class NormalizedMessage:
    content: Optional[str]
    tool_calls: Optional[list[NormalizedToolCall]]
    _anthropic_blocks: Optional[list] = None  # stored for re-insertion on next turn


@dataclass
class NormalizedResponse:
    message: NormalizedMessage


# ── Provider client ───────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, timeout: int = 1800):
        self._timeout = timeout
        self._provider = PROVIDER
        self._init_backend()

    def _init_backend(self):
        if self._provider == "ollama":
            import ollama
            self._backend = ollama.Client(host=OLLAMA_BASE_URL, timeout=self._timeout)
        elif self._provider in ("openai", "fireworks"):
            from openai import OpenAI
            if self._provider == "fireworks":
                self._backend = OpenAI(
                    api_key=os.getenv("FIREWORKS_API_KEY"),
                    base_url="https://api.fireworks.ai/inference/v1",
                    timeout=self._timeout,
                )
            else:
                self._backend = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    timeout=self._timeout,
                )
        elif self._provider == "anthropic":
            import anthropic
            self._backend = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
        else:
            raise ValueError(f"Unknown PROVIDER: {self._provider!r}. Choose: ollama, openai, anthropic, fireworks")

    def chat(
        self,
        model: str,
        messages: list,
        tools: Optional[list] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        num_ctx: Optional[int] = None,   # Ollama only — ignored for cloud providers
        keep_alive: Optional[Union[int, str]] = KEEP_ALIVE,  # Ollama only — ignored for cloud providers
    ) -> NormalizedResponse:
        if self._provider == "ollama":
            return self._chat_ollama(model, messages, tools, temperature, max_tokens, num_ctx, keep_alive)
        elif self._provider in ("openai", "fireworks"):
            return self._chat_openai(model, messages, tools, temperature, max_tokens)
        elif self._provider == "anthropic":
            return self._chat_anthropic(model, messages, tools, temperature, max_tokens)

    # ── Ollama ────────────────────────────────────────────────────────────────

    def _chat_ollama(self, model, messages, tools, temperature, max_tokens, num_ctx, keep_alive) -> NormalizedResponse:
        options: dict = {"temperature": temperature}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        resp = self._backend.chat(
            model=model,
            messages=messages,
            tools=tools or [],
            stream=False,
            options=options,
            keep_alive=keep_alive,
        )
        raw = resp.message
        tool_calls = None
        if raw.tool_calls:
            tool_calls = [
                NormalizedToolCall(
                    function=NormalizedToolFunction(
                        name=tc.function.name,
                        arguments=tc.function.arguments,  # already dict
                    )
                )
                for tc in raw.tool_calls
            ]
        return NormalizedResponse(
            message=NormalizedMessage(content=raw.content or None, tool_calls=tool_calls)
        )

    # ── OpenAI / Fireworks ────────────────────────────────────────────────────

    @staticmethod
    def _convert_messages_for_openai(messages: list) -> list:
        out = []
        for m in messages:
            role = m.get("role")
            if role == "tool":
                entry = {k: v for k, v in m.items() if k != "tool_use_id"}
                if "tool_use_id" in m:
                    entry["tool_call_id"] = m["tool_use_id"]
                out.append(entry)
            elif role == "assistant":
                out.append({k: v for k, v in m.items() if k != "_anthropic_blocks"})
            else:
                out.append(m)
        return out

    def _chat_openai(self, model, messages, tools, temperature, max_tokens) -> NormalizedResponse:
        messages = self._convert_messages_for_openai(messages)
        kwargs: dict = dict(model=model, messages=messages, temperature=temperature)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        for attempt in range(3):
            try:
                resp = self._backend.chat.completions.create(**kwargs)
                break
            except Exception as e:
                # Retry on transient RPM rate limits; re-raise on token-size errors
                if "rate_limit" in type(e).__name__.lower() or "429" in str(e):
                    if "tokens" in str(e).lower() and "reduce" in str(e).lower():
                        raise  # request is too large — retrying won't help
                    wait = 10 * (2 ** attempt)
                    print(f"[providers] rate limit, retrying in {wait}s...")
                    time.sleep(wait)
                    if attempt == 2:
                        raise
                else:
                    raise
        choice = resp.choices[0].message
        tool_calls = None
        if choice.tool_calls:
            tool_calls = [
                NormalizedToolCall(
                    function=NormalizedToolFunction(
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    ),
                    tool_use_id=tc.id,
                )
                for tc in choice.tool_calls
            ]
        return NormalizedResponse(
            message=NormalizedMessage(content=choice.content, tool_calls=tool_calls)
        )

    # ── Anthropic ─────────────────────────────────────────────────────────────

    @staticmethod
    def _convert_tools_for_anthropic(tools: list) -> list:
        out = []
        for t in tools:
            fn = t["function"]
            out.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn["parameters"],
            })
        return out

    @staticmethod
    def _convert_messages_for_anthropic(messages: list) -> tuple[Optional[str], list]:
        """
        Returns (system_prompt, anthropic_messages).
        - Extracts system message as a separate string.
        - Re-inserts stored _anthropic_blocks for assistant turns (preserves tool_use blocks).
        - Converts {"role":"tool",...} entries into tool_result content blocks on a user turn.
        """
        system: Optional[str] = None
        out: list = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                system = m["content"]
                continue
            if role == "assistant":
                # If we stored raw Anthropic blocks, use them to preserve tool_use IDs
                blocks = m.get("_anthropic_blocks")
                if blocks is not None:
                    out.append({"role": "assistant", "content": blocks})
                else:
                    out.append({"role": "assistant", "content": m.get("content", "")})
                continue
            if role == "tool":
                # Attach as a tool_result content block on a user turn
                block = {
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_use_id", "unknown"),
                    "content": m.get("content", ""),
                }
                if out and out[-1]["role"] == "user":
                    existing = out[-1]["content"]
                    if isinstance(existing, str):
                        existing = [{"type": "text", "text": existing}]
                    existing.append(block)
                    out[-1]["content"] = existing
                else:
                    out.append({"role": "user", "content": [block]})
                continue
            # user messages
            out.append({"role": role, "content": m.get("content", "")})
        return system, out

    def _chat_anthropic(self, model, messages, tools, temperature, max_tokens) -> NormalizedResponse:
        system, anthropic_msgs = self._convert_messages_for_anthropic(messages)
        kwargs: dict = dict(
            model=model,
            messages=anthropic_msgs,
            temperature=temperature,
            max_tokens=max_tokens or 8192,
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._convert_tools_for_anthropic(tools)
        resp = self._backend.messages.create(**kwargs)

        content_text: Optional[str] = None
        tool_calls: Optional[list[NormalizedToolCall]] = None
        raw_blocks = resp.content

        for block in raw_blocks:
            if block.type == "text":
                content_text = block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    NormalizedToolCall(
                        function=NormalizedToolFunction(
                            name=block.name,
                            arguments=block.input,  # already dict
                        ),
                        tool_use_id=block.id,
                    )
                )

        return NormalizedResponse(
            message=NormalizedMessage(
                content=content_text,
                tool_calls=tool_calls,
                _anthropic_blocks=raw_blocks,
            )
        )
