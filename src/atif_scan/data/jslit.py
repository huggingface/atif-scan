"""Static reading of tool calls inside a JavaScript tool program (Codex CLI code mode).

Codex's code mode records one `exec` call whose `input` is a program such as
`const r = await tools.exec_command({cmd: "git show HEAD:a.py"}); text(r);`. This module
finds each `tools.NAME({...})` call in code (not in strings or comments) and parses its
argument when it is a *literal* (objects, arrays, strings, numbers, true/false/null), or a
name bound once to a string literal (`const patch = "*** Begin Patch…"; tools.apply_patch
(patch)`). A call inside a loop over a fixed list (`for (const id of ids)`,
`ids.map(id => …)`, with `ids` a never-changed list of strings and integers) is read once
per element, with `${id}`, `"a" + id` and `{id}` filled in: a superset of what ran.
Nothing is executed. Any other value (a reassigned or shadowed name, other `${}`
interpolation, an expression) becomes `UNREAD`, and any other use of `tools` (`tools[k]()`,
`const t = tools`) adds a `("?", UNREAD)` call, so callers treat it as unknown, not absent.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, field, replace
from itertools import product
from math import prod
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


# A template hole naming one variable: `${name}`.
_HOLE = re.compile(r"\$\{\s*([A-Za-z_$][\w$]*)\s*\}")


def _joinable(value: object) -> bool:
    """A string or integer: what `+` and `${}` turn into text exactly as written here.
    Floats (JavaScript prints 2.0 as "2") and booleans are left unread."""
    return isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


def _join(left: object, right: object) -> object:
    """JavaScript's `left + right` for strings and integers: a sum or a concatenation."""
    if isinstance(left, int) and isinstance(right, int):
        return left + right
    return f"{left}{right}"


# A name's value from where it is bound: (that position, the value).
Binding = tuple[int, object]
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


def _without_comments(text: str, lexed: Lexed) -> str:
    """Whitespace in place of comments, preserving offsets and line boundaries.
    Structural regexes must see `x /* comment */ = y` as an assignment, while the
    literal reader keeps the original strings, templates and source positions."""
    starts, ends, _ = lexed
    pieces: list[str] = []
    after = 0
    for start, end in zip(starts, ends, strict=True):
        if text.startswith(("//", "/*"), start):
            pieces.extend((text[after:start], re.sub(r"[^\r\n]", " ", text[start:end])))
            after = end
    pieces.append(text[after:])
    return "".join(pieces)


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
    def __init__(self, text: str, lexed: Lexed, bindings: dict[str, Binding]) -> None:
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
                hole = self.hole()
                if hole is not None:
                    out.append(hole)
                    continue
                readable = False  # interpolation: the runtime value isn't in the source
            out.append(ch)
            self.pos += 1
        raise ValueError("unterminated string")

    def hole(self) -> str | None:
        """A template's `${name}` hole at `pos` when `name` is bound here to a string or
        integer (its text, as JavaScript writes it); moves past it. Else None."""
        match = _HOLE.match(self.text, self.pos)
        value = self.bound(match.group(1)) if match else UNREAD
        if match is None or not _joinable(value):
            return None
        self.pos = match.end()
        return str(value)

    def bound(self, name: str) -> object:
        """The value `name` is bound to at `pos`, else UNREAD."""
        binding = self.bindings.get(name)
        return binding[1] if binding is not None and binding[0] <= self.pos else UNREAD

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
        if value is not _EXPRESSION and self.peek() == "+":
            value = self.concat(value)
        # `"a" + b.c`, `x.join(" ")`, ternaries...: not a plain literal.
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

    def concat(self, first: object) -> object:
        """`first + "b" + name …` from `pos`: strings, integers and bound names, joined
        left to right as JavaScript does; _EXPRESSION for anything else."""
        value = first
        while self.peek() == "+" and not self.text.startswith(("++", "+="), self.pos):
            self.pos += 1
            operand = self.string() if self.peek() in QUOTES else self.scalar()
            if not (_joinable(value) and _joinable(operand)):
                return _EXPRESSION
            value = _join(value, operand)
        return value

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
        value = _KEYWORDS[name] if name in _KEYWORDS else self.bound(name)
        if value is UNREAD:
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
            out[key] = self.bound(key)  # shorthand `{cmd}`: a variable, read when bound


_DECLARE = re.compile(r"(?<![\w$.])(?:const|let|var)(?![\w$])")
_ASSIGN = re.compile(
    r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*(?:\)\s*)*(?:\*\*|>>>|<<|>>|\?\?|\|\||&&|[-+*/%&|^])?=(?![=>])"
)
# Anything else that (re)binds a name: parameters, catch, function/class names, ++/--.
_SHADOW = re.compile(
    r"\b(?:function\b\s*\*?\s*([A-Za-z_$][\w$]*)?\s*|catch\s*)\(([^)]*)"
    r"|\(([^()]*)\)\s*=>|(?<![\w$])([A-Za-z_$][\w$]*)\s*=>"
    r"|(?<![\w$.])(?!(?:if|for|while|switch|with)\b)[A-Za-z_$][\w$]*\s*\(([^()]*)\)\s*\{"
    r"|\bclass\s+([A-Za-z_$][\w$]*)"
    r"|(?<![\w$])([A-Za-z_$][\w$]*)\s*(?:\)\s*)*(?:\+\+|--)"
    r"|(?:\+\+|--|\bdelete\b)\s*(?:\(\s*)*([A-Za-z_$][\w$]*)"
    r"|\bfor\s*(?:await\s*)?\(\s*([A-Za-z_$][\w$]*)\s+(?:of|in)\b"
)
_DESTRUCTURE = re.compile(r"[\]}]\s*=(?![=>])")
_END_LITERAL = re.compile(r"[ \t]*(?:[;,\n})]|$)")


@dataclass
class _Declarations:
    declared: Counter[str] = field(default_factory=Counter)  # declarations per name
    values: dict[str, tuple[int, object]] = field(default_factory=dict)  # (after, value)
    shadowed: set[str] = field(default_factory=set)  # destructured names
    sites: dict[str, int] = field(default_factory=dict)  # where each name is declared


def _bindings(program: str, reader: _Reader) -> tuple[dict[str, Binding], dict[str, Binding]]:
    """Names declared exactly once and never reassigned, destructured, or used as a
    parameter: (those bound to a string literal, those bound to a list of strings and
    integers that is only ever read). Strings are immutable and such a list is never
    changed, so the name holds that literal wherever it's in scope. One pass each."""
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
    once = {
        name: (pos, value)
        for name, (pos, value) in found.values.items()
        if found.declared[name] == 1 and assigned[name] == 1 and name not in shadowed
    }
    strings = {name: b for name, b in once.items() if isinstance(b[1], str)}
    candidates = {
        name: value
        for name, (_, value) in once.items()
        if isinstance(value, list)
        and 0 < len(value) <= MAX_EXPANSION
        and all(_joinable(v) for v in value)
    }
    uses = _uses(program, reader, set(candidates))
    lists: dict[str, Binding] = {
        name: (once[name][0], tuple(value))
        for name, value in candidates.items()
        if all(at == found.sites[name] or _only_reads(program, reader, at) for at in uses[name])
    }
    return strings, lists


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
        found.sites[name] = ident.start()
    else:
        return False
    after, value = _initializer(program, reader)
    if name:
        found.values[name] = (after, value)
    return True


def _initializer(program: str, reader: _Reader) -> tuple[int, object]:
    """(position after the value, the value when it's a string or array literal, else
    UNREAD)."""
    after = reader.pos
    if reader.peek() != "=" or program[reader.pos + 1 : reader.pos + 2] in "=>":
        return after, UNREAD
    reader.pos += 1
    ch, start = reader.peek(), reader.pos
    try:
        value = reader.string() if ch in QUOTES else reader.literal(0) if ch == "[" else UNREAD
    except (ValueError, IndexError, KeyError, RecursionError):
        reader.pos, value = start, UNREAD  # not a literal: skipped as an expression below
    if not _END_LITERAL.match(program, reader.pos):
        value = UNREAD  # `"a" + b`, `"x".repeat(3)`: an expression
    after = reader.pos
    reader.expression()
    return after, value


def _rebound(program: str, reader: _Reader, start: int = 0, end: int | None = None) -> set[str]:
    """Names rebound in code (between `start` and `end`) other than by assignment:
    parameters, catch, function/class names, ++/--, destructuring assignment."""
    end = len(program) if end is None else end
    shadowed: set[str] = set()
    for m in _SHADOW.finditer(program, start, end):
        if reader.code(m.start()):
            for group in filter(None, m.groups()):
                shadowed.update(_IDENT.findall(group))
    for m in _DESTRUCTURE.finditer(program, start, end):  # `[a, b] = x`, `({a} = x)`
        if reader.code(m.start()) and m.start() in reader.pairs:
            shadowed.update(_IDENT.findall(program, reader.pairs[m.start()], m.start()))
    return shadowed


# Loops over a list bound in the program: a call inside runs once per element, so it is
# read once per element (a superset of what ran: `break` or an `if` only run fewer).
MAX_EXPANSION = 64  # element combinations read per call before leaving it unread
# `const`/`let` only: a `var` outlives the loop, so a callback made in it could see it
# reassigned afterwards.
_FOR_OF = re.compile(r"(?<![\w$.])for\s*\(\s*(?:const|let)\s+([A-Za-z_$][\w$]*)\s+of(?![\w$])")
# Code that can rebind names out of sight: no loop is read in a program using it.
_DYNAMIC = re.compile(r"(?<![\w$.])(?:eval|Function|with)\s*\(")
_EACH = re.compile(r"\.\s*(?:map|forEach|flatMap)\s*\(")
_CALLBACK = re.compile(
    r"\s*(?:async\s+)?(?:([A-Za-z_$][\w$]*)"
    r"|\(\s*([A-Za-z_$][\w$]*)\s*(?:,\s*[A-Za-z_$][\w$]*\s*)?\))\s*=>"
)
_NAME_BEFORE = re.compile(r"(?<![\w$.])([A-Za-z_$][\w$]*)\Z")
_NAME_USE = re.compile(r"(?<![\w$.])[A-Za-z_$][\w$]*")
_CLOSE_NEXT = re.compile(r"\s*\)")
LOOKBACK = 256  # characters searched back for a name before `.map(` or `of`
MAX_LOOPS = 200  # loop forms in one program before none is read (keeps reading linear)
# Uses of a list that leave it as it is: these methods, `.length`, `xs[i]` reads.
_READ_USE = re.compile(
    r"\s*(?:\.\s*(?:(map|forEach|flatMap|filter|some|every|find|findIndex)\s*\("
    r"|(?:includes|indexOf|join|slice|at)\s*\(|(length)(?![\w$]))|(\[))"
)
_WRITE = re.compile(
    r"\s*(?:\)\s*)*(?:(?:\*\*|>>>|<<|>>|\?\?|\|\||&&|[-+*/%&|^])?=(?!=)"
    r"|\+\+|--|(?:of|in)\b)"
)
_OF_BEFORE = re.compile(r"(?<![\w$])of\s*\Z")


def _back(program: str, pos: int) -> int:
    """Where the code before `pos` ends, past trailing whitespace (bounded by LOOKBACK)."""
    stop = max(0, pos - LOOKBACK)
    while pos > stop and program[pos - 1].isspace():
        pos -= 1
    return pos


def _uses(program: str, reader: _Reader, names: set[str]) -> dict[str, list[int]]:
    """Where each of `names` occurs in code, in one pass."""
    found: dict[str, list[int]] = {name: [] for name in names}
    if names:
        for m in _NAME_USE.finditer(program):
            if m.group() in found and reader.code(m.start()):
                found[m.group()].append(m.start())
    return found


def _only_reads(program: str, reader: _Reader, at: int) -> bool:
    """The use of a list name at `at` (not its declaration) only reads the list: a loop's
    source, a non-mutating method, `.length` or an element read. Passing it anywhere else
    (a function could change it) is not."""
    name = _NAME_USE.match(program, at)
    if name is None:
        return False
    after = name.end()
    of = _OF_BEFORE.search(program, max(0, at - LOOKBACK), at)
    return bool(of and _CLOSE_NEXT.match(program, after)) or _reads(program, reader, after)


def _reads(program: str, reader: _Reader, end: int) -> bool:
    """The use of a list name ending at `end` only reads it."""
    use = _READ_USE.match(program, end)
    if use is None:
        return False
    if use.group(1) is not None:
        # A callback is passed the list itself as its third argument: allow only an
        # inline arrow taking at most two.
        return _CALLBACK.match(program, use.end()) is not None
    if use.group(3) is not None:
        close = reader.pairs.get(use.start(3))
        return close is not None and not _WRITE.match(program, close + 1)
    return use.group(2) is None or not _WRITE.match(program, use.end())


@dataclass(frozen=True)
class _Loop:
    start: int  # the body: where the element is bound
    end: int
    name: str
    values: tuple[object, ...]
    uses: tuple[int, ...] = ()  # where `name` occurs in code in the body, ascending

    def used_in(self, start: int, end: int) -> bool:
        k = bisect_left(self.uses, start)
        return k < len(self.uses) and self.uses[k] < end


def _loops(program: str, reader: _Reader, lists: dict[str, Binding]) -> list[_Loop]:
    """`for (const x of SOURCE) BODY` and `SOURCE.map(x => BODY)` (also forEach,
    flatMap) whose SOURCE is a list literal or a bound list and whose BODY never rebinds
    `x`, outermost first."""
    forms = [
        (match, read)
        for pattern, read in ((_FOR_OF, _for_of), (_EACH, _each))
        for match in pattern.finditer(program)
        if reader.code(match.start())
    ]
    if len(forms) > MAX_LOOPS:
        return []
    found: list[_Loop] = []
    for match, read in forms:
        try:
            loop = read(program, reader, lists, match)
            if loop is not None and loop.name not in _rebound_in(program, reader, loop):
                found.append(loop)
        except (ValueError, IndexError, KeyError, RecursionError):
            continue
    uses = _uses(program, reader, {loop.name for loop in found})
    return sorted(
        (
            replace(loop, uses=tuple(u for u in uses[loop.name] if loop.start <= u < loop.end))
            for loop in found
        ),
        key=lambda loop: loop.start,
    )


def _source(reader: _Reader, lists: dict[str, Binding], name: str | None) -> object:
    """The list a loop runs over: the bound list `name`, or the literal at `reader.pos`."""
    if name is not None:
        binding = lists.get(name)
        return binding[1] if binding is not None and binding[0] <= reader.pos else None
    value = reader.literal(0) if reader.peek() == "[" else None
    ok = isinstance(value, list) and 0 < len(value) <= MAX_EXPANSION
    return tuple(value) if ok and all(_joinable(v) for v in value) else None


def _for_of(
    program: str, reader: _Reader, lists: dict[str, Binding], match: re.Match[str]
) -> _Loop | None:
    close = reader.pairs[program.index("(", match.start())]
    reader.pos = match.end()
    reader.skip()
    ident = _IDENT.match(program, reader.pos)
    if reader.peek() == "[":
        values = _source(reader, lists, None)
    elif ident is not None:
        reader.pos = ident.end()
        values = _source(reader, lists, ident.group())
    else:
        return None
    if not isinstance(values, tuple) or reader.peek() != ")" or reader.pos != close:
        return None
    reader.pos = close + 1
    if reader.peek() == "{":
        return _Loop(reader.pos, reader.pairs[reader.pos], match.group(1), values)
    start = reader.pos
    reader.expression()
    return _Loop(start, reader.pos, match.group(1), values)


def _each(
    program: str, reader: _Reader, lists: dict[str, Binding], match: re.Match[str]
) -> _Loop | None:
    values = _each_source(program, reader, lists, _back(program, match.start()))
    callback = _CALLBACK.match(program, match.end())
    if not isinstance(values, tuple) or callback is None:
        return None
    end = reader.pairs[match.end() - 1]
    return _Loop(callback.end(), end, callback.group(1) or callback.group(2), values)


def _each_source(program: str, reader: _Reader, lists: dict[str, Binding], end: int) -> object:
    """The list the code before `end` ends with, for `.map(…)` after it: a list literal or
    a bound list name; None for anything else."""
    if end and program[end - 1] == "]":
        opening = reader.pairs[end - 1]
        previous = _back(program, opening)
        named = _NAME_BEFORE.search(program, max(0, previous - LOOKBACK), previous)
        if named or (previous and program[previous - 1] in ")]"):
            return None  # `xs[0].map(…)`, `f()[1].map(…)`: an element, not a literal
        reader.pos = opening
        values = _source(reader, lists, None)
        return values if reader.pos == end else None
    if named := _NAME_BEFORE.search(program, max(0, end - LOOKBACK), end):
        reader.pos = named.end()
        return _source(reader, lists, named.group(1))
    return None


def _rebound_in(program: str, reader: _Reader, loop: _Loop) -> set[str]:
    """Names a loop's body declares, assigns or rebinds."""
    assigned: set[str]
    found = _Declarations()
    for match in _DECLARE.finditer(program, loop.start, loop.end):
        if reader.code(match.start()):
            reader.pos = match.end()
            _declaration(program, reader, found)
    assigned = {
        m[1] for m in _ASSIGN.finditer(program, loop.start, loop.end) if reader.code(m.start())
    }
    return (
        set(found.declared)
        | found.shadowed
        | assigned
        | _rebound(program, reader, loop.start, loop.end)
    )


def _expanded(reader: _Reader, start: int, loops: list[_Loop]) -> list[object]:
    """The argument of the call whose `(` ends at `start`, once per combination of the
    elements of the loops around it that it uses; one UNREAD when there are too many."""
    base = reader.bindings
    first, end = _argument(reader, start, base)
    around = [loop for loop in loops if loop.start <= start < loop.end and loop.used_in(start, end)]
    if not around:
        return [first]
    if prod(len(loop.values) for loop in around) > MAX_EXPANSION:
        return [UNREAD]
    out: dict[str, object] = {}
    for combination in product(*(loop.values for loop in around)):
        bound = dict(base)
        for loop, value in zip(around, combination, strict=True):
            bound[loop.name] = (loop.start, value)  # inner loops last: they win
        value = _argument(reader, start, bound)[0]
        out.setdefault(repr(value), value)
    return list(out.values())


def _argument(reader: _Reader, start: int, bindings: dict[str, Binding]) -> tuple[object, int]:
    """(the argument at `start` read with these bindings, where it ends)."""
    saved, reader.bindings, reader.pos = reader.bindings, bindings, start
    try:
        value = None if reader.peek() == ")" else reader.value()
    except (ValueError, IndexError, KeyError, RecursionError):
        value = UNREAD
    finally:
        reader.bindings = saved
    return value, reader.pos


def tool_calls(program: str) -> list[tuple[str, object]]:
    """(tool name, argument) for each `tools.NAME(arg)` call in code, in source order. The
    argument is the parsed literal, UNREAD when it isn't one, or None when the call has
    none; ("?", UNREAD) marks any other use of `tools`. Never raises: unreadable is UNREAD."""
    try:
        lexed = _lex(program)
        reader = _Reader(program, lexed, {})
        syntax = _without_comments(program, lexed)
        reader.bindings, lists = _bindings(syntax, reader)
        dynamic = any(reader.code(m.start()) for m in _DYNAMIC.finditer(syntax))
        loops = [] if dynamic else _loops(syntax, reader, lists)
    except (ValueError, IndexError, KeyError, RecursionError):
        return [("?", UNREAD)] if CALL.search(program) else []
    calls: list[tuple[str, object]] = []
    for match in CALL.finditer(program):
        if not reader.code(match.start()):
            continue  # inside a string, template text, comment or regex literal
        if match.group(1) is None:
            calls.append(("?", UNREAD))  # `tools[k](…)`, `tools.x?.(…)`, `const t = tools`
            continue
        calls.extend((match.group(1), a) for a in _expanded(reader, match.end(), loops))
    if not calls and CALL.search(program):
        calls.append(("?", UNREAD))  # `tools` only seen outside code: unsure, so unknown
    return calls


# Codex code mode's own state helpers: `store(key, value)` keeps a value between programs
# and `load(key)` reads it back. The value lives in the harness, not the trace (TB4
# production-planning: 17 table dumps stored, then loaded and printed in parts).
STATE_HELPERS = ("store", "load")
_STATE_CALL = re.compile(r"(?<![\w$.])(store|load)\s*\(")
_STATE_DECLARED = re.compile(
    r"(?<![\w$.])(?:function\s*\*?|const|let|var|class)\s+(store|load)(?![\w$])"
)


def state_helper_calls(program: str) -> frozenset[str]:
    """The harness state helpers (`STATE_HELPERS`) this program calls in code (not in a
    string or comment). A name the program declares itself is its own function, not the
    harness's. Unlexable programs are searched as text, so a use isn't missed."""
    try:
        reader = _Reader(program, _lex(program), {})
        in_code = reader.code
    except (ValueError, IndexError, KeyError, RecursionError):

        def in_code(_: int) -> bool:
            return True

    own: set[str] = {m.group(1) for m in _STATE_DECLARED.finditer(program) if in_code(m.start())}
    used: set[str] = {m.group(1) for m in _STATE_CALL.finditer(program) if in_code(m.start())}
    return frozenset(used - own)


# Patch text is file content, read through the apply_patch call it feeds (programs build
# patches from `+`-joined pieces: any piece with an envelope line).
_PATCH = re.compile(r"^\*\*\* (?:Begin Patch|End Patch|(?:Add|Update|Delete) File:|Move to:)", re.M)
# A lone word ("set", "env", an object key or method name) is code, not a command or
# path: kept are literals with a space, `/`, `.`, `:` or `-` (commands, paths, URLs,
# flags, hyphenated names), judged without the space a `${}` hole leaves. TB4
# circuit-fibsqrt: an error message `set ${i}` read as `set`, the shell builtin that
# prints the environment.
_WORDY = re.compile(r"\S[\s/.:-]|[/.:-]\S")


def program_strings(program: str) -> list[str]:
    """The program's string and template literals, decoded, in source order: the constants
    a computed argument is built from (`const cmds = [...]` run by a loop, a template's
    text). One item per literal, so each is read like an argument string (JavaScript
    quoting isn't shell quoting). A template's `${…}` holes are a space. Left out: text a
    call's argument already holds (read as that call's fields), patch text, comments,
    regex literals and lone words. Nothing is executed."""
    read = {text for _, argument in tool_calls(program) for text in _strings(argument)}
    starts, ends, _ = _lex(program)
    out: list[str] = []
    template: list[str] = []
    for start, end in zip(starts, ends, strict=True):
        parts = _literal_parts(program[start:end], template)
        if parts is None:
            continue
        text = _decode("".join(parts))
        if text not in read and _WORDY.search(text.strip()) and not _PATCH.search(text):
            out.append(text)
    return out


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _literal_parts(raw: str, template: list[str]) -> list[str] | None:
    """A string literal's text, or a whole template's parts once its closing "`" is
    reached (`template` collects the chunks before it); None for anything else."""
    if raw[0] in ("'", '"'):
        return [raw[1:-1] if len(raw) > 1 and raw[-1] == raw[0] else raw[1:]]
    if raw[0] not in ("`", "}"):
        return None  # a comment or regex literal
    if raw[0] == "`":
        template.clear()
    body = raw[1:]
    if body.endswith("${"):
        template.extend((body[:-2], " "))
        return None
    parts = [*template, body[:-1] if body.endswith("`") else body]
    template.clear()
    return parts


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
