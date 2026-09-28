"""Static reading of tool calls inside a JavaScript tool program (Codex CLI code mode).

Codex's code mode records one `exec` call whose `input` is a program such as
`const r = await tools.exec_command({cmd: "git show HEAD:a.py"}); text(r);`. This module
finds each `tools.NAME({...})` call and parses its argument when it is a *literal*
(objects, arrays, strings, numbers, true/false/null), or a name bound once to a string
literal (`const patch = "*** Begin Patch…"; tools.apply_patch(patch)`). Nothing is
executed. Any other value (a reassigned or shadowed name, `${}` interpolation, an
expression) becomes `UNREAD`, so callers can treat it as unknown evidence, not absent.
"""

from __future__ import annotations

import re

UNREAD = object()  # a non-literal value: present but not statically readable
CALL = re.compile(r"\btools\.([A-Za-z_]\w*)\s*\(")
_IDENT = re.compile(r"[A-Za-z_$][\w$]*")
_NUMBER = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}


_DECLARED = re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=(?![=>])")


def _string_bindings(program: str) -> dict[str, tuple[int, str]]:
    """Names bound exactly once to a string literal and never reassigned or used as a
    parameter (which could shadow them): name -> (position after the binding, value).
    Strings are immutable, so such a name holds that literal wherever it's in scope."""
    found: dict[str, tuple[int, object]] = {}
    for match in _DECLARED.finditer(program):
        reader = _Reader(program, match.end(), {})
        try:
            value = reader.string() if reader.peek() in "\"'`" else UNREAD
        except ValueError:
            value = UNREAD
        if not re.match(r"[ \t]*(?:[;,\n]|$)", program[reader.pos :]):
            value = UNREAD  # `"a" + b`, `"x".repeat(3)`: an expression, not the literal
        name = match.group(1)
        found[name] = (reader.pos, UNREAD if name in found else value)
    bindings = {}
    for name, (pos, value) in found.items():
        n = re.escape(name)
        assigned = re.findall(rf"(?<![\w$.]){n}\s*(?:[-+*/%]|\?\?|\|\||&&)?=(?![=>])", program)
        parameter = re.search(
            rf"\bfunction\b[^(]*\([^)]*(?<![\w$]){n}(?![\w$])|"
            rf"\([^()]*(?<![\w$]){n}(?![\w$])[^()]*\)\s*=>|(?<![\w$.]){n}\s*=>",
            program,
        )
        if isinstance(value, str) and len(assigned) == 1 and not parameter:
            bindings[name] = (pos, value)
    return bindings


class _Reader:
    def __init__(self, text: str, pos: int, bindings: dict[str, tuple[int, str]]):
        self.text, self.pos, self.bindings = text, pos, bindings

    def skip(self) -> None:
        text = self.text
        while self.pos < len(text):
            if text[self.pos].isspace():
                self.pos += 1
            elif text.startswith("//", self.pos):
                end = text.find("\n", self.pos)
                self.pos = len(text) if end < 0 else end
            elif text.startswith("/*", self.pos):
                end = text.find("*/", self.pos + 2)
                self.pos = len(text) if end < 0 else end + 2
            else:
                return

    def peek(self) -> str:
        self.skip()
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def string(self):
        quote = self.text[self.pos]
        self.pos += 1
        out = []
        readable = True
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch == "\\":
                nxt = self.text[self.pos + 1 : self.pos + 2]
                if nxt == "u" and re.match(
                    r"[0-9a-fA-F]{4}", self.text[self.pos + 2 : self.pos + 6]
                ):
                    out.append(chr(int(self.text[self.pos + 2 : self.pos + 6], 16)))
                    self.pos += 6
                    continue
                out.append(_ESCAPES.get(nxt, nxt))
                self.pos += 2
                continue
            if ch == quote:
                self.pos += 1
                return "".join(out) if readable else UNREAD
            if quote == "`" and self.text.startswith("${", self.pos):
                readable = False  # interpolation: the runtime value isn't in the source
            out.append(ch)
            self.pos += 1
        raise ValueError("unterminated string")

    def value(self, depth: int = 0):
        if depth > 50:
            raise ValueError("too deep")
        ch = self.peek()
        if ch in "\"'`":
            value = self.string()
        elif ch == "{":
            value = self.obj(depth)
        elif ch == "[":
            value = self.array(depth)
        else:
            number = _NUMBER.match(self.text, self.pos)
            ident = _IDENT.match(self.text, self.pos)
            if number:
                self.pos = number.end()
                text = number.group()
                value = float(text) if any(c in text for c in ".eE") else int(text)
            elif ident and ident.group() in ("true", "false", "null", "undefined"):
                self.pos = ident.end()
                value = {"true": True, "false": False}.get(ident.group())
            elif ident and self.bindings.get(ident.group(), (len(self.text),))[0] <= self.pos:
                self.pos = ident.end()
                value = self.bindings[ident.group()][1]
            else:
                self.expression()
                return UNREAD
        # `"a" + b`, `x.join(" ")`, ternaries...: not a plain literal.
        if self.peek() not in (",", "}", "]", ")", ""):
            self.expression()
            return UNREAD
        return value

    def expression(self) -> None:
        """Skip to the end of the current value (bracket- and string-aware)."""
        depth = 0
        while self.pos < len(self.text):
            ch = self.peek()
            if ch in "\"'`":
                self.string()
                continue
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth == 0:
                    return
                depth -= 1
            elif ch == "," and depth == 0:
                return
            self.pos += 1

    def obj(self, depth: int):
        self.pos += 1
        out: dict = {}
        while True:
            start = self.pos
            ch = self.peek()
            if ch == "":
                raise ValueError("unterminated object")
            if ch == "}":
                self.pos += 1
                return out
            if ch in "\"'":
                key = self.string()
            elif ch == "." and self.text.startswith("...", self.pos):
                self.pos += 3
                self.value(depth + 1)
                out["..."] = UNREAD  # spread: unknown extra keys
                key = None
            else:
                ident = _IDENT.match(self.text, self.pos)
                if not ident:
                    raise ValueError("bad key")
                key = ident.group()
                self.pos = ident.end()
            if key is not None:
                if self.peek() == ":":
                    self.pos += 1
                    out[key if isinstance(key, str) else "?"] = self.value(depth + 1)
                else:
                    out[key] = UNREAD  # shorthand `{cmd}`: a variable
            if self.peek() == ",":
                self.pos += 1
            elif self.peek() != "}" or self.pos == start:
                raise ValueError("bad object")

    def array(self, depth: int):
        self.pos += 1
        out = []
        while True:
            start = self.pos
            ch = self.peek()
            if ch == "":
                raise ValueError("unterminated array")
            if ch == "]":
                self.pos += 1
                return out
            out.append(self.value(depth + 1))
            if self.peek() == ",":
                self.pos += 1
            elif self.peek() != "]" or self.pos == start:
                raise ValueError("bad array")


def tool_calls(program: str) -> list[tuple[str, object]]:
    """(tool name, argument) for each `tools.NAME(arg)` call, in source order. The argument
    is the parsed literal, UNREAD when it isn't one, or None when the call has none."""
    calls = []
    bindings = _string_bindings(program)
    for match in CALL.finditer(program):
        reader = _Reader(program, match.end(), bindings)
        try:
            argument = None if reader.peek() == ")" else reader.value()
        except ValueError:
            argument = UNREAD
        calls.append((match.group(1), argument))
    return calls
