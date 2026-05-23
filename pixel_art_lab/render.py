"""JSON -> PNG renderer. Zero dependencies (hand-rolled PNG encoder).

Usage:
    python3 render.py <spec.json> <out.png>

Spec format:
{
  "width": 32, "height": 32, "scale": 16,
  "background": "#bfe9ff",
  "palette": { ".": null, "B": "#2b2b2b", ... },   # null => background
  "rows": [ "....", ... ]   # each string length == width, count == height
}
"""

import json
import struct
import sys
import zlib


def hex_to_rgb(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def write_png(path, width, height, pixel_rows):
    raw = bytearray()
    for row in pixel_rows:
        raw.append(0)  # filter type: none
        for (r, g, b) in row:
            raw += bytes((r, g, b))

    def chunk(typ, data):
        return (
            struct.pack(">I", len(data))
            + typ
            + data
            + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
        )

    out = bytearray(b"\x89PNG\r\n\x1a\n")
    out += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    out += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    out += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(out)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "cat.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "out.png"

    with open(src) as f:
        spec = json.load(f)

    W, H = spec["width"], spec["height"]
    scale = spec.get("scale", 1)
    bg = hex_to_rgb(spec.get("background", "#ffffff"))

    palette = {}
    for k, v in spec["palette"].items():
        palette[k] = bg if v is None else hex_to_rgb(v)

    rows = spec["rows"]

    errs = []
    if len(rows) != H:
        errs.append(f"row count {len(rows)} != height {H}")
    for i, r in enumerate(rows):
        if len(r) != W:
            errs.append(f"row {i:>2}: len {len(r)} != width {W}")
        bad = sorted(set(ch for ch in r if ch not in palette))
        if bad:
            errs.append(f"row {i:>2}: unknown chars {bad}")
    if errs:
        print("SPEC ERRORS:")
        for e in errs:
            print("  -", e)
        sys.exit(1)

    native = [[palette[ch] for ch in r] for r in rows]

    big = []
    for prow in native:
        big_row = []
        for px in prow:
            big_row.extend([px] * scale)
        for _ in range(scale):
            big.append(big_row)

    write_png(dst, W * scale, H * scale, big)
    print(f"OK: wrote {dst} ({W * scale}x{H * scale}) from {src}")


if __name__ == "__main__":
    main()
