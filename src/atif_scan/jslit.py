"""Static reading of tool calls inside a JavaScript tool program (Codex CLI code mode).

Codex's code mode records one `exec` call whose `input` is a program such as
`const r = await tools.exec_command({cmd: "git show HEAD:a.py"}); text(r);`. This module
finds each `tools.NAME({...})` call in code (not in strings or comments) and parses its
argument when it is a *literal* (objects, arrays, strings, numbers, true/false/null), or a
name bound once to a string literal (`const patch = "*** Begin Patch…"; tools.apply_patch
(patch)`). Nothing is executed. Any other value (a reassigned or shadowed name, `${}`
interpolation, an expression) becomes `UNREAD`, and any other use of `tools` (`tools[k]()`,
`const t = tools`) adds a `("?", UNREAD)` call, so callers treat it as unknown, not absent.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter

UNREAD = object()  # a non-literal value: present but not statically readable
# Any reference to `tools`; group 1 is the name when it is the plain `tools.NAME(` form.
CALL = re.compile(r"(?<![\w$])tools(?![\w$])(?:\s*\.\s*([A-Za-z_$][\w$]*)\s*\()?")
QUOTES = ('"', "'", "`")
_IDENT = re.compile(r"[A-Za-z_$][\w$]*")
_NUMBER = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}
_CONTINUATION = ("\n", "\r", "\u2028", "\u2029")  # backslash-newline: nothing
_KEYWORDS = {"true": True, "false": False, "null": None, "undefined": None}
_HEX = re.compile(r"x([0-9a-fA-F]{2})|u([0-9a-fA-F]{4})|u\{([0-9a-fA-F]{1,6})\}")

# Lexer: strings, template text, comments and regex literals are not code.
_TOKEN = re.compile(r"//|/\*|[\"'`/(){}\[\]]")
_STRING = {q: re.compile(rf"{q}(?:\\[\s\S]|[^{q}\\\n])*{q}?") for q in "\"'"}
_TEMPLATE = re.compile(r"(?:\\[\s\S]|[^`\\$]|\$(?!\{))*(`|\$\{)?")
_REGEX = re.compile(r"/(?:\\.|\[(?:\\.|[^\]\\\n])*\]|[^/\\\n\[])+/")
_REGEX_AFTER = frozenset("(,=:[!&|?{};+-*%<>~^")  # else `/` is division (stays code)
_PAIRS = {")": "(", "]": "[", "}": "{"}


def _lex(text: str) -> tuple[list[int], list[int], dict[int, int]]:
    """One pass: sorted non-code (start, end) spans and matched bracket positions (both
    ways). Unsure whether `/` starts a regex: treat it as code, so calls are still seen."""
    starts, ends, pairs, stack = [], [], {}, []  # stack: bracket positions, None = `${`
    pos = 0
    while match := _TOKEN.search(text, pos):
        tok, i = match.group(), match.start()
        end = match.end()
        if tok in ("//", "/*"):
            close = text.find("\n" if tok == "//" else "*/", end)
            end = len(text) if close < 0 else close + (0 if tok == "//" else 2)
        elif tok in ("'", '"'):
            end = _STRING[tok].match(text, i).end()
        elif tok == "`" or (tok == "}" and stack and stack[-1] is None):
            if tok == "}":
                stack.pop()
            chunk = _TEMPLATE.match(text, end)
            end = chunk.end()
            if chunk.group(1) == "${":
                stack.append(None)
        elif tok == "/":
            j = i - 1
            while j >= 0 and text[j].isspace():
                j -= 1
            regex = _REGEX.match(text, i) if j < 0 or text[j] in _REGEX_AFTER else None
            if not regex:
                pos = end
                continue
            end = regex.end()
        else:
            if tok in "([{":
                stack.append(i)
            elif stack and stack[-1] is not None and text[stack[-1]] == _PAIRS[tok]:
                pairs[stack[-1]] = i
                pairs[i] = stack.pop()
            pos = end
            continue
        starts.append(i)
        ends.append(end)
        pos = end
    return starts, ends, pairs


class _Reader:
    def __init__(self, text: str, lexed, bindings: dict[str, tuple[int, str]]):
        self.text, self.pos, self.bindings = text, 0, bindings
        self.starts, self.ends, self.pairs = lexed

    def code(self, pos: int) -> bool:
        k = bisect_right(self.starts, pos) - 1
        return k < 0 or pos >= self.ends[k]

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
        text, quote = self.text, self.text[self.pos]
        self.pos += 1
        out, readable = [], True
        while self.pos < len(text):
            ch = text[self.pos]
            if ch == "\\":
                nxt = text[self.pos + 1 : self.pos + 2]
                code = _HEX.match(text, self.pos + 1)
                if code:
                    point = int(next(g for g in code.groups() if g), 16)
                    out.append(chr(point) if point < 0x110000 else "\ufffd")
                    self.pos = code.end()
                    continue
                if text.startswith("\r\n", self.pos + 1):
                    nxt = "\r\n"
                out.append("" if nxt[:1] in _CONTINUATION else _ESCAPES.get(nxt, nxt))
                self.pos += 1 + max(len(nxt), 1)
                continue
            if ch == quote:
                self.pos += 1
                return "".join(out) if readable else UNREAD
            if quote != "`" and ch == "\n":
                break  # quoted strings end at the line
            if quote == "`" and text.startswith("${", self.pos):
                readable = False  # interpolation: the runtime value isn't in the source
            out.append(ch)
            self.pos += 1
        raise ValueError("unterminated string")

    def value(self, depth: int = 0):
        if depth > 50:
            raise ValueError("too deep")
        ch = self.peek()
        number = _NUMBER.match(self.text, self.pos)
        ident = _IDENT.match(self.text, self.pos)
        name = ident.group() if ident else None
        if ch in QUOTES:
            value = self.string()
        elif ch == "{":
            value = self.items("}", lambda out: self.entry(out, depth), {})
        elif ch == "[":
            value = self.items("]", lambda out: out.append(self.value(depth + 1)), [])
        elif number:
            self.pos, text = number.end(), number.group()
            value = float(text) if any(c in text for c in ".eE") else int(text)
        elif name in _KEYWORDS:
            self.pos, value = ident.end(), _KEYWORDS[name]
        elif name in self.bindings and self.bindings[name][0] <= self.pos:
            self.pos, value = ident.end(), self.bindings[name][1]
        else:
            self.expression()
            return UNREAD
        # `"a" + b`, `x.join(" ")`, ternaries...: not a plain literal.
        if self.peek() not in (",", "}", "]", ")", ""):
            self.expression()
            return UNREAD
        return value

    def expression(self) -> None:
        """Skip to the end of the current value, jumping over matched brackets."""
        while ch := self.peek():
            if ch in QUOTES:
                self.string()
                continue
            if ch in "([{":
                if self.pos not in self.pairs:
                    raise ValueError("unbalanced")
                self.pos = self.pairs[self.pos]
            elif ch in ")]},;":
                return
            self.pos += 1

    def items(self, close: str, item, out):
        self.pos += 1
        while (ch := self.peek()) != close:
            start = self.pos
            if not ch:
                raise ValueError("unterminated")
            item(out)
            if self.peek() == ",":
                self.pos += 1
            elif self.peek() != close or self.pos == start:
                raise ValueError("bad literal")
        self.pos += 1
        return out

    def entry(self, out: dict, depth: int) -> None:
        if self.peek() in ("'", '"'):
            key = self.string()
        elif self.text.startswith("...", self.pos):
            self.pos += 3
            self.value(depth + 1)
            out["..."] = UNREAD  # spread: unknown extra keys
            return
        elif ident := _IDENT.match(self.text, self.pos):
            key, self.pos = ident.group(), ident.end()
        else:
            raise ValueError("bad key")  # includes computed `[k]: v`
        if self.peek() == ":":
            self.pos += 1
            out[key] = self.value(depth + 1)
        else:
            out[key] = UNREAD  # shorthand `{cmd}`: a variable


_DECLARE = re.compile(r"(?<![\w$.])(?:const|let|var)(?![\w$])")
_ASSIGN = re.compile(
    r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*(?:\*\*|>>>|<<|>>|\?\?|\|\||&&|[-+*/%&|^])?=(?![=>])"
)
# Anything else that (re)binds a name: parameters, catch, function/class names, ++/--.
_SHADOW = re.compile(
    r"\b(?:function\b\s*\*?\s*([A-Za-z_$][\w$]*)?\s*|catch\s*)\(([^)]*)"
    r"|\(([^()]*)\)\s*=>|(?<![\w$])([A-Za-z_$][\w$]*)\s*=>"
    r"|(?<![\w$.])(?!(?:if|for|while|switch|with)\b)[A-Za-z_$][\w$]*\s*\(([^()]*)\)\s*\{"
    r"|\bclass\s+([A-Za-z_$][\w$]*)"
    r"|(?<![\w$])([A-Za-z_$][\w$]*)\s*(?:\+\+|--)|(?:\+\+|--)\s*([A-Za-z_$][\w$]*)"
)
_DESTRUCTURE = re.compile(r"[\]}]\s*=(?![=>])")
_END_LITERAL = re.compile(r"[ \t]*(?:[;,\n})]|$)")


def _string_bindings(program: str, reader: _Reader) -> dict[str, tuple[int, str]]:
    """Names declared exactly once, to a string literal, and never reassigned, destructured,
    or used as a parameter: name -> (position after the binding, value). Strings are
    immutable, so such a name holds that literal wherever it's in scope. One pass each."""
    declared: Counter[str] = Counter()
    values: dict[str, tuple[int, object]] = {}
    shadowed: set[str] = set()
    for match in _DECLARE.finditer(program):
        if not reader.code(match.start()):
            continue
        reader.pos = match.end()
        try:
            while True:
                if reader.peek() in ("{", "["):  # destructuring: never a literal
                    end = reader.pairs[reader.pos]
                    shadowed.update(_IDENT.findall(program, reader.pos, end))
                    reader.pos, name = end + 1, None
                elif ident := _IDENT.match(program, reader.pos):
                    reader.pos, name = ident.end(), ident.group()
                    declared[name] += 1
                else:
                    break
                value, after = UNREAD, reader.pos
                if reader.peek() == "=" and program[reader.pos + 1 : reader.pos + 2] not in "=>":
                    reader.pos += 1
                    if reader.peek() in QUOTES:
                        value = reader.string()
                    if not _END_LITERAL.match(program, reader.pos):
                        value = UNREAD  # `"a" + b`, `"x".repeat(3)`: an expression
                    after = reader.pos
                    reader.expression()
                if name:
                    values[name] = (after, value)
                if reader.peek() != ",":
                    break
                reader.pos += 1
        except (ValueError, IndexError, KeyError):
            continue
    assigned = Counter(m[1] for m in _ASSIGN.finditer(program) if reader.code(m.start()))
    for m in _SHADOW.finditer(program):
        if reader.code(m.start()):
            for group in filter(None, m.groups()):
                shadowed.update(_IDENT.findall(group))
    for m in _DESTRUCTURE.finditer(program):  # `[a, b] = x`, `({a} = x)`
        if reader.code(m.start()) and m.start() in reader.pairs:
            shadowed.update(_IDENT.findall(program, reader.pairs[m.start()], m.start()))
    return {
        name: (pos, value)
        for name, (pos, value) in values.items()
        if isinstance(value, str)
        and declared[name] == 1
        and assigned[name] == 1
        and name not in shadowed
    }


def tool_calls(program: str) -> list[tuple[str, object]]:
    """(tool name, argument) for each `tools.NAME(arg)` call in code, in source order. The
    argument is the parsed literal, UNREAD when it isn't one, or None when the call has
    none; ("?", UNREAD) marks any other use of `tools`. Never raises: unreadable is UNREAD."""
    try:
        reader = _Reader(program, _lex(program), {})
        reader.bindings = _string_bindings(program, reader)
    except (ValueError, IndexError, KeyError, RecursionError):
        return [("?", UNREAD)] if CALL.search(program) else []
    calls = []
    for match in CALL.finditer(program):
        if not reader.code(match.start()):
            continue  # inside a string, template text, comment or regex literal
        if match.group(1) is None:
            calls.append(("?", UNREAD))  # `tools[k](…)`, `tools.x?.(…)`, `const t = tools`
            continue
        reader.pos = match.end()
        try:
            argument = None if reader.peek() == ")" else reader.value()
        except (ValueError, IndexError, KeyError, RecursionError):
            argument = UNREAD
        calls.append((match.group(1), argument))
    if not calls and CALL.search(program):
        calls.append(("?", UNREAD))  # `tools` only seen outside code: unsure, so unknown
    return calls
