"""Screenshot the static viewer and check it isn't visibly broken, before a release.

    uv run python tools/viewer_shots.py                  # a synthetic export, all sizes
    uv run python tools/viewer_shots.py --viewer DIR     # an existing (private) export
    uv run python tools/viewer_shots.py --chrome /path/to/chrome --out DIR

Exports `examples/synthetic.json` (or uses `--viewer DIR`), screenshots it with headless
Chrome at desktop and phone widths in light and dark themes, and checks each image:
not blank, content from the top (a page that opens scrolled hid the masthead once,
and a phone layout rendered blank), and the theme actually applied. Screenshots are
written for a person (or a vision model) to look at; this script never calls a model.
After fast-agent's docs/scripts/docs_visual_assess.py. Exit 1 on any failed check.

A real export holds masked trace text: its screenshots are as private as the export, so
the default output is the atif-scan home (`reviews/`), never the source tree.
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from atif_scan.data import paths

ROOT = Path(__file__).resolve().parent.parent
SIZES = {"desktop": (1440, 1000), "phone": (390, 900)}
TOP_BAND = 80  # rows that must hold something: the page starts at its masthead
MIN_INK = 0.02  # share of non-background pixels below which a shot is blank
BRIGHT, DARK = 0.6, 0.35  # mean luminance a light / dark theme must clear
SAMPLE = 3  # check every third pixel and row: plenty for these thresholds
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CHANNELS = {2: 3, 6: 4}  # PNG colour type -> bytes per pixel (8-bit RGB, RGBA)
BIT_DEPTH = 8


@dataclass(frozen=True)
class Image:
    width: int
    height: int
    channels: int
    rows: list[bytes]

    def pixel(self, x: int, y: int) -> tuple[int, int, int]:
        at = x * self.channels
        r, g, b = self.rows[y][at : at + 3]
        return r, g, b


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else b if pb <= pc else c


def _unfilter(kind: int, line: bytearray, prior: bytes, bpp: int) -> bytes:
    """Undo one PNG scanline filter (RFC 2083 6.1)."""
    for i in range(len(line)):
        left = line[i - bpp] if i >= bpp else 0
        up = prior[i]
        corner = prior[i - bpp] if i >= bpp else 0
        add = (0, left, up, (left + up) // 2, _paeth(left, up, corner))[kind]
        line[i] = (line[i] + add) & 0xFF
    return bytes(line)


def read_png(path: Path) -> Image:
    """An 8-bit, non-interlaced RGB(A) PNG, as Chrome writes them."""
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG")
    at, header, idat = len(PNG_SIGNATURE), b"", bytearray()
    while at < len(data):
        (size,) = struct.unpack(">I", data[at : at + 4])
        kind, body = data[at + 4 : at + 8], data[at + 8 : at + 8 + size]
        header = body if kind == b"IHDR" else header
        idat += body if kind == b"IDAT" else b""
        at += size + 12
    width, height, depth, colour, _, _, interlace = struct.unpack(">IIBBBBB", header)
    if depth != BIT_DEPTH or colour not in CHANNELS or interlace:
        raise ValueError("unsupported PNG layout")
    bpp, raw = CHANNELS[colour], zlib.decompress(bytes(idat))
    stride, rows, prior = width * bpp, [], bytes(width * bpp)
    for y in range(height):
        start = y * (stride + 1)
        prior = _unfilter(raw[start], bytearray(raw[start + 1 : start + 1 + stride]), prior, bpp)
        rows.append(prior)
    return Image(width, height, bpp, rows)


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255


def problems(image: Image, dark: bool) -> list[str]:
    """What's visibly wrong with one screenshot (empty when nothing is)."""
    background = image.pixel(1, 1)
    ys = range(0, image.height, SAMPLE)
    xs = range(0, image.width, SAMPLE)
    inked = [y for y in ys if any(image.pixel(x, y) != background for x in xs)]
    found = []
    if len(inked) < MIN_INK * len(ys):
        found.append("blank: almost nothing but background")
    elif inked[0] >= TOP_BAND:
        found.append(f"nothing in the top {TOP_BAND}px: the page doesn't start at its masthead")
    mean = sum(_luminance(image.pixel(x, y)) for y in ys for x in xs) / (len(ys) * len(xs))
    if dark and mean > DARK:
        found.append(f"dark theme not applied (mean luminance {mean:.2f})")
    if not dark and mean < BRIGHT:
        found.append(f"light theme not applied (mean luminance {mean:.2f})")
    return found


def chrome(explicit: str | None) -> str:
    for candidate in (explicit, os.environ.get("CHROME"), "google-chrome", "chromium"):
        found = candidate and shutil.which(candidate)
        if found:
            return found
    sys.exit("viewer_shots: no Chrome found; pass --chrome PATH or set $CHROME")


def shoot(browser: str, page: Path, out: Path, size: tuple[int, int], dark: bool) -> None:
    flags = ["--force-dark-mode"] if dark else []
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run(
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--user-data-dir={profile}",
                "--virtual-time-budget=5000",
                f"--window-size={size[0]},{size[1]}",
                f"--screenshot={out}",
                *flags,
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )


def export_synthetic(directory: Path) -> Path:
    viewer = directory / "viewer"
    example = str(ROOT / "examples/synthetic.json")
    subprocess.run(
        [sys.executable, "-m", "atif_scan", example, "--viewer", str(viewer), "--format", "json"],
        check=True,
        capture_output=True,
    )
    return viewer


def run(viewer: Path, out: Path, browser: str) -> int:
    failed = 0
    for name, size in SIZES.items():
        for dark in (False, True):
            shot = out / f"{name}-{'dark' if dark else 'light'}.png"
            shoot(browser, viewer / "index.html", shot, size, dark)
            found = problems(read_png(shot), dark) if shot.is_file() else ["no screenshot written"]
            failed += bool(found)
            notes = "".join(f"\n     {p}" for p in found)
            print(f"{'FAIL' if found else 'ok  '} {shot.name}{notes}")
    print(f"screenshots: {out}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--viewer", type=Path, help="an existing export (default: synthetic)")
    parser.add_argument("--out", type=Path, help="screenshot folder (default: atif-scan home)")
    parser.add_argument("--chrome", help="Chrome or Chromium executable")
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or paths.home() / "reviews" / f"viewer-shots-{stamp}"
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    browser = chrome(args.chrome)
    if args.viewer and not (args.viewer / "index.html").is_file():
        sys.exit(f"viewer_shots: no index.html in {args.viewer} (is it a finished export?)")
    if args.viewer:
        return run(args.viewer, out, browser)
    with tempfile.TemporaryDirectory() as scratch:
        return run(export_synthetic(Path(scratch)), out, browser)


if __name__ == "__main__":
    sys.exit(main())
