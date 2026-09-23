"""ATIF v1 adapter. Reject invalid structure; preserve unknown text representations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import MappingProxyType

from .model import Channel, Content, Observation, Step, ToolCall, Trace

MAX_BYTES = 64 * 1024 * 1024


class TraceError(ValueError):
    """Messages contain fixed error codes only, never input snippets or filenames."""


def content(value: object) -> Content:
    if value is None:
        return Content()
    if isinstance(value, str):
        return Content(value)
    if isinstance(value, list):
        parts = [content(part) for part in value]
        return Content("\n".join(p.text for p in parts), all(p.understood for p in parts))
    if isinstance(value, dict):
        # Only text-bearing blocks, never arbitrary metadata/URL/image payloads.
        if value.get("type") in ("image", "image_url", "audio", "video"):
            return Content()
        if isinstance(value.get("text"), str):
            return Content(value["text"])
    return Content(understood=False)


def normalize_tool(name: str) -> str:
    aliases = {
        "bash": "shell",
        "Shell": "shell",
        "shell": "shell",
        "read_text_file": "read_text_file",
        "web_search": "web_search",
        "web_search_preview": "web_search",
        "web_fetch": "web_fetch",
        "fetch_url": "web_fetch",
    }
    return aliases.get(name, "other")


def freeze(value: object) -> object:
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise TraceError("invalid_argument_keys")
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TraceError("invalid_argument_value")


def parse_trace(value: object) -> Trace:
    if not isinstance(value, dict) or not isinstance(value.get("steps"), list):
        raise TraceError("invalid_steps")
    version = value.get("schema_version")
    if version is not None and (
        not isinstance(version, str) or re.fullmatch(r"ATIF-v1\.\d+(?:\.\d+)?", version) is None
    ):
        raise TraceError("unsupported_schema_version")
    steps = []
    call_ids: set[str] = set()
    for index, raw in enumerate(value["steps"]):
        if not isinstance(raw, dict) or raw.get("source") not in ("system", "user", "agent"):
            raise TraceError("invalid_step")
        copied = raw.get("is_copied_context", False)
        if not isinstance(copied, bool):
            raise TraceError("invalid_copied_context")
        raw_calls = raw.get("tool_calls")
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list):
            raise TraceError("invalid_calls")
        calls = []
        for call_index, call in enumerate(raw_calls):
            if not isinstance(call, dict) or not isinstance(call.get("function_name"), str):
                raise TraceError("invalid_call")
            call_id = call.get("tool_call_id")
            if not isinstance(call_id, str):
                raise TraceError("invalid_call_id")
            if not copied:
                if call_id in call_ids:
                    raise TraceError("duplicate_call_id")
                call_ids.add(call_id)
            args = call.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = None
            tool = normalize_tool(call["function_name"])
            channels = {
                "shell": (Channel.COMMAND,),
                "read_text_file": (Channel.PATH,),
                "web_search": (Channel.QUERY,),
                "web_fetch": (Channel.URL,),
            }.get(tool, ())
            fields = tuple(
                (
                    channel,
                    content(args[channel.value])
                    if isinstance(args, dict) and isinstance(args.get(channel.value), str)
                    else Content(understood=False),
                )
                for channel in channels
            )
            calls.append(
                ToolCall(
                    call_index,
                    tool,
                    fields,
                    call_id,
                    call["function_name"],
                    freeze(args) if isinstance(args, dict) else None,
                )
            )
        observation = raw.get("observation")
        if observation is None:
            observation = {}
        if not isinstance(observation, dict) or not isinstance(
            observation.get("results", []), list
        ):
            raise TraceError("invalid_observation")
        observations = []
        for item in observation.get("results", []):
            if not isinstance(item, dict):
                raise TraceError("invalid_observation_result")
            source_call_id = item.get("source_call_id")
            if source_call_id is not None and not isinstance(source_call_id, str):
                raise TraceError("invalid_observation_call_id")
            observations.append(Observation(source_call_id, content(item.get("content"))))
        steps.append(
            Step(
                index,
                raw["source"],
                copied,
                content(raw.get("message")),
                content(raw.get("reasoning_content")),
                tuple(calls),
                tuple(observations),
            )
        )
    return Trace(version, tuple(steps))


def load_trace(path: Path) -> Trace:
    try:
        if path.stat().st_size > MAX_BYTES:
            raise TraceError("trace_too_large")
        return parse_trace(json.loads(path.read_text(encoding="utf-8")))
    except TraceError:
        raise
    except (OSError, UnicodeError, ValueError, RecursionError):
        raise TraceError("unreadable_trace") from None
