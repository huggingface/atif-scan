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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

UNREAD = object()  # a non-literal value: present but not statically readable
_EXPRESSION = object()  # not the start of a literal (so the value is UNREAD)
UNICODE_END = 0x110000  # past the last code point
MAX_NESTING = 50  # literal nesting depth read before giving up (unreadable)
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


# (non-code span starts, their ends, matched bracket positions both ways)
Lexed = tuple[list[int], list[int], dict[int, int]]


def _lex(text: str) -> Lexed:
    """One pass: sorted non-code (start, end) spans and matched bracket positions (both
    ways). Unsure whether `/` starts a regex: treat it as code, so calls are still seen."""
    lexer = _Lexer(text)
    pos = 0
    while match := _TOKEN.search(text, pos):
        pos = lexer.token(match)
    return lexer.starts, lexer.ends, lexer.pairs


def _end(pattern: re.Pattern[str], text: str, pos: int) -> int:
    """Where `pattern` (one that always matches at `pos`) stops."""
    match = pattern.match(text, pos)
    return match.end() if match else pos


class _Lexer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.pairs: dict[int, int] = {}
        self.stack: list[int | None] = []  # open bracket positions, None = `${`

    def token(self, match: re.Match[str]) -> int:
        """Record one token; the position to search on from."""
        tok, i = match.group(), match.start()
        in_template = bool(self.stack) and self.stack[-1] is None
        if tok in "()[]{}" and not (tok == "}" and in_template):
            self.bracket(tok, i)
            return match.end()
        end = self.not_code(tok, i, match.end())
        if end is None:  # a `/` that is division
            return match.end()
        self.starts.append(i)
        self.ends.append(end)
        return end

    def bracket(self, tok: str, i: int) -> None:
        stack = self.stack
        if tok in "([{":
            stack.append(i)
        elif stack and (top := stack[-1]) is not None and self.text[top] == _PAIRS[tok]:
            self.pairs[top] = i
            self.pairs[i] = top
            stack.pop()

    def not_code(self, tok: str, i: int, end: int) -> int | None:
        """The end of the comment, string, template text or regex literal starting at `i`
        (None when a `/` is division)."""
        text = self.text
        if tok in ("//", "/*"):
            close = text.find("\n" if tok == "//" else "*/", end)
            return len(text) if close < 0 else close + (0 if tok == "//" else 2)
        if tok in ("'", '"'):
            return _end(_STRING[tok], text, i)
        return self.regex(i) if tok == "/" else self.template(tok, end)

    def template(self, tok: str, end: int) -> int:
        """After "`" or the "}" closing a `${`: template text up to its end or next `${`."""
        if tok == "}":
            self.stack.pop()
        chunk = _TEMPLATE.match(self.text, end)
        if chunk is None:  # never: the pattern matches the empty string
            return end
        if chunk.group(1) == "${":
            self.stack.append(None)
        return chunk.end()

    def regex(self, i: int) -> int | None:
        text = self.text
        j = i - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        regex = _REGEX.match(text, i) if j < 0 or text[j] in _REGEX_AFTER else None
        return regex.end() if regex else None


class _Reader:
    def __init__(self, text: str, lexed: Lexed, bindings: dict[str, tuple[int, str]]) -> None:
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

    def string(self) -> object:
        """The string or template literal at `pos` (UNREAD when it interpolates)."""
        text, readable = self.quoted()
        return text if readable else UNREAD

    def quoted(self) -> tuple[str, bool]:
        """(text, readable) of the string or template literal at `pos`."""
        text, quote = self.text, self.text[self.pos]
        self.pos += 1
        out: list[str] = []
        readable = True
        while self.pos < len(text):
            ch = text[self.pos]
            if ch == "\\":
                out.append(self.escape())
                continue
            if ch == quote:
                self.pos += 1
                return "".join(out), readable
            if quote != "`" and ch == "\n":
                break  # quoted strings end at the line
            if quote == "`" and text.startswith("${", self.pos):
                readable = False  # interpolation: the runtime value isn't in the source
            out.append(ch)
            self.pos += 1
        raise ValueError("unterminated string")

    def escape(self) -> str:
        """What the backslash escape at `pos` stands for; moves past it."""
        text = self.text
        code = _HEX.match(text, self.pos + 1)
        if code:
            point = int(next(g for g in code.groups() if g), 16)
            self.pos = code.end()
            return chr(point) if point < UNICODE_END else "\ufffd"
        nxt = "\r\n" if text.startswith("\r\n", self.pos + 1) else text[self.pos + 1 : self.pos + 2]
        self.pos += 1 + max(len(nxt), 1)
        return "" if nxt[:1] in _CONTINUATION else _ESCAPES.get(nxt, nxt)

    def value(self, depth: int = 0) -> object:
        if depth > MAX_NESTING:
            raise ValueError("too deep")
        value = self.literal(depth)
        # `"a" + b`, `x.join(" ")`, ternaries...: not a plain literal.
        if value is _EXPRESSION or self.peek() not in (",", "}", "]", ")", ""):
            self.expression()
            return UNREAD
        return value

    def literal(self, depth: int) -> object:
        """The literal at `pos`, or _EXPRESSION when it doesn't start one."""
        ch = self.peek()
        if ch in QUOTES:
            return self.string()
        if ch == "{":
            entries: dict[str, object] = {}
            self.items("}", lambda: self.entry(entries, depth))
            return entries
        if ch == "[":
            elements: list[object] = []
            self.items("]", lambda: elements.append(self.value(depth + 1)))
            return elements
        return self.scalar()

    def scalar(self) -> object:
        """A number, keyword or bound name at `pos`, else _EXPRESSION."""
        number = _NUMBER.match(self.text, self.pos)
        if number:
            self.pos, text = number.end(), number.group()
            return float(text) if any(c in text for c in ".eE") else int(text)
        ident = _IDENT.match(self.text, self.pos)
        name = ident.group() if ident else None
        if ident is None or name is None:
            return _EXPRESSION
        if name in _KEYWORDS:
            value = _KEYWORDS[name]
        elif name in self.bindings and self.bindings[name][0] <= self.pos:
            value = self.bindings[name][1]
        else:
            return _EXPRESSION
        self.pos = ident.end()
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

    def items(self, close: str, item: Callable[[], None]) -> None:
        """Read `item`s separated by commas up to `close`."""
        self.pos += 1
        while (ch := self.peek()) != close:
            start = self.pos
            if not ch:
                raise ValueError("unterminated")
            item()
            if self.peek() == ",":
                self.pos += 1
            elif self.peek() != close or self.pos == start:
                raise ValueError("bad literal")
        self.pos += 1

    def entry(self, out: dict[str, object], depth: int) -> None:
        if self.peek() in ("'", '"'):
            key = self.quoted()[0]  # always readable: only templates interpolate
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


@dataclass
class _Declarations:
    declared: Counter[str] = field(default_factory=Counter)  # declarations per name
    values: dict[str, tuple[int, object]] = field(default_factory=dict)  # (after, value)
    shadowed: set[str] = field(default_factory=set)  # destructured names


def _string_bindings(program: str, reader: _Reader) -> dict[str, tuple[int, str]]:
    """Names declared exactly once, to a string literal, and never reassigned, destructured,
    or used as a parameter: name -> (position after the binding, value). Strings are
    immutable, so such a name holds that literal wherever it's in scope. One pass each."""
    found = _Declarations()
    for match in _DECLARE.finditer(program):
        if not reader.code(match.start()):
            continue
        reader.pos = match.end()
        try:
            _declaration(program, reader, found)
        except (ValueError, IndexError, KeyError):
            continue
    assigned = Counter(m[1] for m in _ASSIGN.finditer(program) if reader.code(m.start()))
    shadowed = found.shadowed | _rebound(program, reader)
    return {
        name: (pos, value)
        for name, (pos, value) in found.values.items()
        if isinstance(value, str)
        and found.declared[name] == 1
        and assigned[name] == 1
        and name not in shadowed
    }


def _declaration(program: str, reader: _Reader, found: _Declarations) -> None:
    """The declarators after `const`/`let`/`var` at `reader.pos` (`a = "x", {b} = y, c`)."""
    while _declarator(program, reader, found):
        if reader.peek() != ",":
            return
        reader.pos += 1


def _declarator(program: str, reader: _Reader, found: _Declarations) -> bool:
    """One declarator at `reader.pos`; False when there is none."""
    if reader.peek() in ("{", "["):  # destructuring: never a literal
        end = reader.pairs[reader.pos]
        found.shadowed.update(_IDENT.findall(program, reader.pos, end))
        reader.pos, name = end + 1, None
    elif ident := _IDENT.match(program, reader.pos):
        reader.pos, name = ident.end(), ident.group()
        found.declared[name] += 1
    else:
        return False
    after, value = _initializer(program, reader)
    if name:
        found.values[name] = (after, value)
    return True


def _initializer(program: str, reader: _Reader) -> tuple[int, object]:
    """(position after the value, the value when it's a string literal, else UNREAD)."""
    after = reader.pos
    if reader.peek() != "=" or program[reader.pos + 1 : reader.pos + 2] in "=>":
        return after, UNREAD
    reader.pos += 1
    value = reader.string() if reader.peek() in QUOTES else UNREAD
    if not _END_LITERAL.match(program, reader.pos):
        value = UNREAD  # `"a" + b`, `"x".repeat(3)`: an expression
    after = reader.pos
    reader.expression()
    return after, value


def _rebound(program: str, reader: _Reader) -> set[str]:
    """Names rebound in code other than by assignment: parameters, catch, function/class
    names, ++/--, destructuring assignment."""
    shadowed: set[str] = set()
    for m in _SHADOW.finditer(program):
        if reader.code(m.start()):
            for group in filter(None, m.groups()):
                shadowed.update(_IDENT.findall(group))
    for m in _DESTRUCTURE.finditer(program):  # `[a, b] = x`, `({a} = x)`
        if reader.code(m.start()) and m.start() in reader.pairs:
            shadowed.update(_IDENT.findall(program, reader.pairs[m.start()], m.start()))
    return shadowed


# A located call: (tool name, argument, span of a fully readable argument or None).
Located = tuple[str, object, tuple[int, int] | None]


def tool_calls(program: str) -> list[tuple[str, object]]:
    """(tool name, argument) for each `tools.NAME(arg)` call in code, in source order. The
    argument is the parsed literal, UNREAD when it isn't one, or None when the call has
    none; ("?", UNREAD) marks any other use of `tools`. Never raises: unreadable is UNREAD."""
    return [(name, argument) for name, argument, _ in _located_calls(program)[0]]


def _located_calls(program: str) -> tuple[list[Located], Lexed | None]:
    try:
        lexed = _lex(program)
        reader = _Reader(program, lexed, {})
        reader.bindings = _string_bindings(program, reader)
    except (ValueError, IndexError, KeyError, RecursionError):
        return ([("?", UNREAD, None)] if CALL.search(program) else []), None
    calls: list[Located] = []
    for match in CALL.finditer(program):
        if not reader.code(match.start()):
            continue  # inside a string, template text, comment or regex literal
        if match.group(1) is None:
            calls.append(("?", UNREAD, None))  # `tools[k](…)`, `tools.x?.(…)`, `t = tools`
            continue
        calls.append((match.group(1), *_argument(reader, match.end())))
    if not calls and CALL.search(program):
        calls.append(("?", UNREAD, None))  # `tools` only seen outside code: unsure
    return calls, lexed


def _argument(reader: _Reader, pos: int) -> tuple[object, tuple[int, int] | None]:
    """The call's argument at `pos`, and its span when every part of it was read."""
    reader.pos = pos
    try:
        if reader.peek() == ")":
            return None, None
        start = reader.pos
        argument = reader.value()
    except (ValueError, IndexError, KeyError, RecursionError):
        return UNREAD, None
    return argument, (start, reader.pos) if _readable(argument) else None


def _readable(value: object) -> bool:
    if value is UNREAD:
        return False
    if isinstance(value, dict):
        return all(_readable(v) for v in value.values())
    if isinstance(value, list):
        return all(_readable(v) for v in value)
    return True


# Patch text is file content, read through the apply_patch call it feeds: any piece of an
# envelope (programs build patches from `+`-joined pieces), or only hunk lines.
_PATCH = re.compile(r"^\*\*\* (?:Begin Patch|End Patch|(?:Add|Update|Delete) File:|Move to:)", re.M)
_HUNK_LINES = re.compile(r"(?:(?:[+\- ][^\n]*|@@[^\n]*)(?:\n|$))+")


_CHANGED_LINE = re.compile(r"^[+-]", re.M)
# A lone word ("set", "env", "utf8", an object key or method name) is code, not a command
# or path: kept are literals with a space, `/`, `.`, `:` or `-` (commands, paths, URLs,
# flags, hyphenated names), judged without the space a `${}` hole leaves. TB4
# circuit-fibsqrt: an error message `set ${i}` read as `set`, the shell builtin that
# prints the environment.
_WORDY = re.compile(r"\S[\s/.:-]|[/.:-]\S")


def _patch_text(text: str) -> bool:
    """Envelope text, or several lines that are all hunk lines with one changed: a single
    line (`" && ls"`, `"--resolve host:443:ip"`) is never taken for a hunk."""
    if _PATCH.search(text):
        return True
    lines = text.strip("\n")
    return "\n" in lines and bool(_HUNK_LINES.fullmatch(text) and _CHANGED_LINE.search(text))


def program_strings(program: str) -> list[str]:
    """The program's string and template literals beyond the tool arguments read as calls,
    decoded, in source order: the constants a computed argument is built from (`const
    cmds = [...]`, a template's text, `+` joins). One item per literal, so each is read
    like an argument string (JavaScript quoting around it is not shell quoting). Left
    out: fully readable `tools.NAME(…)` arguments (their calls' fields), apply_patch
    envelopes (file content), comments and regex literals. A template's `${…}` holes are
    a space. Nothing is executed."""
    calls, lexed = _located_calls(program)
    if lexed is None:
        return []
    read = sorted(span for _, _, span in calls if span is not None)
    out: list[str] = []
    template: list[str] = []
    for start, end in zip(*lexed[:2], strict=True):
        parts = _literal_parts(program[start:end], template)
        if parts is None or _inside(read, start):
            continue
        text = _decode("".join(parts))
        if _WORDY.search(text.strip()) and not _patch_text(text):
            out.append(text)
    return out


def _literal_parts(raw: str, template: list[str]) -> list[str] | None:
    """A string literal's text, or a whole template's once its last chunk is reached;
    None for comments, regex literals and template chunks before the last."""
    if raw[0] in ("'", '"'):
        return [raw[1:-1] if len(raw) > 1 and raw[-1] == raw[0] else raw[1:]]
    if raw[0] in ("`", "}"):
        parts, done = _template_chunk(raw, template)
        return parts if done else None
    return None


def _template_chunk(raw: str, template: list[str]) -> tuple[list[str], bool]:
    """Collect one template text chunk (after "`" or a hole's "}"); the whole template's
    parts and True once its closing "`" is reached."""
    if raw[0] == "`":
        template.clear()
    body = raw[1:]
    if body.endswith("${"):
        template.extend((body[:-2], " "))
        return [], False
    template.append(body[:-1] if body.endswith("`") else body)
    parts = list(template)
    template.clear()
    return parts, True


def _inside(spans: list[tuple[int, int]], pos: int) -> bool:
    k = bisect_right(spans, (pos, UNICODE_END)) - 1
    return k >= 0 and spans[k][0] <= pos < spans[k][1]


def _decode(text: str) -> str:
    """Backslash escapes in string or template text, as `_Reader.escape` reads them."""
    if "\\" not in text:
        return text
    reader = _Reader(text, ([], [], {}), {})
    out: list[str] = []
    while reader.pos < len(text):
        if text[reader.pos] == "\\":
            out.append(reader.escape())
        else:
            out.append(text[reader.pos])
            reader.pos += 1
    return "".join(out)
