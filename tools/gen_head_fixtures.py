#!/usr/bin/env python3
"""
tools/gen_head_fixtures.py

Writes the folded SR-head oracle fixtures used by extension/tests/e2e/gpu_parity.spec.ts.

The expected values are produced independently of the extension code, with the research
emulator's own half helpers (h/hb from semantics/emulator/scalar.py) and the exact
reproduction of semantics/emulator/check_folded_heads.py, whose formula matches the
statically evaluated ea1940 SASS on all 147,456 half values:

  * source patterns black / irregular ramp / corner+interior impulse on a 32x32 half4
    source I = [H(2R-1), H(2G-1), H(2B-1), 0];
  * learned bias modes: zero, and the fixture bias1.bin (z = H((k-24)/128));
  * baseline coefficients read from each fixture's constant bank at 0x628;
  * out = H(H(B + T) - 0.5), T = H((1 + tanh z) / 2), planar RGB 64x64.

Both folded heads are covered: head-folded-low-fixture (low, C16) and head-folded-fixture
(medium, C32). The research directory is only read. Output goes to
extension/tests/fixtures/head/. The script is deterministic and idempotent: a second run
rewrites byte-identical files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path


def research_root() -> Path:
    """Directory of the RTX VSR research dataset, from RTX_VSR_RESEARCH_DIR (not part of this repository)."""
    value = os.environ.get("RTX_VSR_RESEARCH_DIR")
    if not value:
        raise SystemExit("Set RTX_VSR_RESEARCH_DIR to the RTX VSR research dataset directory.")
    path = Path(value).expanduser()
    if not path.is_dir():
        raise SystemExit(f"RTX_VSR_RESEARCH_DIR is not a directory: {path}")
    return path


RESEARCH = research_root()
SEMANTICS = RESEARCH / "semantics"
sys.path.insert(0, str(SEMANTICS / "emulator"))
from scalar import h, hb  # noqa: E402  (research emulator helpers)

OUT_DIR = Path(__file__).resolve().parent.parent / "extension" / "tests" / "fixtures" / "head"
FIXTURES = {"low": ("head-folded-low-fixture", 16), "medium": ("head-folded-fixture", 32)}
KINDS = ["black", "ramp", "impulse"]
SIZE = 32
OUT = 64


def H(x: float) -> float:
    return h(hb(x))


def D(v, w):
    return H(H(v[2] * w[2] + H(v[0] * w[0])) + H(v[3] * w[3] + H(v[1] * w[1])))


def baseline(pix, c, y, x, coefs):
    fx = (x + 0.5) / 2 - 0.5
    fy = (y + 0.5) / 2 - 0.5
    ix = math.floor(fx)
    iy = math.floor(fy)
    rows = []
    for dy in (-1, 0, 1, 2):
        taps = [pix[max(0, min(SIZE - 1, iy + dy)), max(0, min(SIZE - 1, ix + dx))][c] for dx in (-1, 0, 1, 2)]
        rows.append(D(taps, coefs[x % 2]))
    return D(rows, coefs[y % 2])


def source_rgb(kind: str, x: int, y: int):
    if kind == "black":
        return [0.0, 0.0, 0.0]
    if kind == "ramp":
        return [x / 31, y / 31, ((x + 3 * y) % 7) / 6]
    return [float(x == 0 and y == 0), float(x == 15 and y == 15), float(x == 31 and y == 31)]


def pack_halves(values) -> bytes:
    return b"".join(struct.pack("<H", hb(v)) for v in values)


def write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    manifest = {
        "generator": "tools/gen_head_fixtures.py",
        "oracle": "semantics/emulator/check_folded_heads.py formula (scalar.h/hb)",
        "source_size": [SIZE, SIZE],
        "output_size": [OUT, OUT],
        "layout": {
            "source": "half4 [2R-1,2G-1,2B-1,0], row-major 32x32, little-endian FP16",
            "bias1": "48 FP16 learned 1x1 biases",
            "expected": "planar RGB FP16, plane 64x64",
        },
        "models": {},
    }
    total = 0
    for model, (folder, channels) in FIXTURES.items():
        fixture_dir = SEMANTICS / folder
        cfg = json.loads((fixture_dir / "fixture.json").read_text())
        cbank = Path(cfg["constant_bank"]).read_bytes()
        coefs = [[h(v) for v in struct.unpack_from("<4H", cbank, 0x628 + 8 * q)] for q in range(2)]
        bias_mem = next(b for b in cfg["memory"] if b["name"] == "bias1")
        stored_bias = (fixture_dir / "bias1.bin").read_bytes()[: bias_mem["logical_bytes"]]
        stored_bias_values = [h(v) for v in struct.unpack("<48H", stored_bias)]
        expected_bias = [H((k - 24) / 128) for k in range(48)]
        if stored_bias_values != expected_bias:
            raise SystemExit(f"{folder}: bias1.bin is not H((k-24)/128)")
        cases = []
        for kind in KINDS:
            for mixed in (False, True):
                bias = stored_bias_values if mixed else [0.0] * 48
                raw = []
                pix = {}
                for y in range(SIZE):
                    for x in range(SIZE):
                        i4 = [H(2 * v - 1) for v in source_rgb(kind, x, y)] + [0.0]
                        raw += i4
                        pix[y, x] = [H(H(v + 1) * 0.5) for v in i4[:3]]
                expected = [0.0] * (3 * OUT * OUT)
                for c in range(3):
                    for y in range(OUT):
                        for x in range(OUT):
                            k = 16 * c + 4 * (y % 4) + x % 4
                            z = bias[k]
                            t = H((1 + math.tanh(z)) / 2)
                            b = baseline(pix, c, y, x, coefs)
                            expected[c * OUT * OUT + y * OUT + x] = H(H(b + t) - 0.5)
                name = f"{kind}-{int(mixed)}"
                base = OUT_DIR / model
                files = {
                    "source": write(base / f"{name}.source.bin", pack_halves(raw)),
                    "bias1": write(base / f"{name}.bias1.bin", pack_halves(bias)),
                    "expected": write(base / f"{name}.expected.bin", pack_halves(expected)),
                }
                total += len(expected)
                cases.append({"name": name, "source": kind, "unique_bias": mixed, "half_values": len(expected), "sha256": files})
        manifest["models"][model] = {
            "research_fixture": folder,
            "feature_channels": channels,
            "baseline_coefficients": coefs,
            "cases": cases,
        }
    manifest["total_half_values"] = total
    write(OUT_DIR / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    print(f"wrote {OUT_DIR} ({total} expected half values)")
    if total != 147_456:
        raise SystemExit(f"expected 147456 half values, produced {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
