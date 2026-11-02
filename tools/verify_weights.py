#!/usr/bin/env python3
"""
tools/verify_weights.py

Independently re-checks the bundles written by convert_weights.py against the
immutable research artifacts:

  * every manifest tensor exists in all-tensors-derived.json and
    all-parameter-registries.json and nothing registered is missing;
  * bundle bytes at each offset equal the raw parameter file and its SHA-256;
  * float16 tensors have length == prod(logical_shape) * 2 and finite values;
  * opaque8 calibration entries (logical_shape None) match the raw file size;
  * offsets are 16-byte aligned, non-overlapping and padding bytes are zero;
  * RB/TB 1x1 skip biases are exactly zero; SR head biases are not all zero;
  * bundle totals equal the expected sizes.
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


RESEARCH_DIR = research_root()
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "extension" / "public" / "weights"

EXPECTED = {
    "low": {"aligned": 884_816, "raw": 884_552, "tensors": 99, "float16": 66, "opaque8": 33,
            "channels": [16, 32, 32, 64, 64]},
    "medium": {"aligned": 1_687_728, "raw": 1_687_464, "tensors": 99, "float16": 66, "opaque8": 33,
               "channels": [32, 64, 64, 64, 64]},
}


class VerificationError(Exception):
    pass


def require(cond: bool, message: str) -> None:
    if not cond:
        raise VerificationError(message)


def verify_model(model: str, derived: dict[str, dict], registry_names: set[str]) -> dict:
    exp = EXPECTED[model]
    manifest = json.loads((WEIGHTS_DIR / f"{model}_manifest.json").read_text())
    blob = (WEIGHTS_DIR / f"{model}.bin").read_bytes()

    require(manifest["model"] == model, f"{model}: manifest model field mismatch")
    require(len(blob) == manifest["total_size_bytes"], f"{model}: bundle size differs from manifest")
    require(len(blob) == exp["aligned"], f"{model}: bundle is {len(blob)} B, expected {exp['aligned']} B")
    require(hashlib.sha256(blob).hexdigest() == manifest["bundle_sha256"], f"{model}: bundle SHA-256 mismatch")
    require(manifest["encoder_channels"] == exp["channels"], f"{model}: encoder channels {manifest['encoder_channels']}")

    tensors = manifest["tensors"]
    require(len(tensors) == exp["tensors"], f"{model}: {len(tensors)} tensors, expected {exp['tensors']}")
    require(set(tensors) == set(derived), f"{model}: manifest/derived tensor name sets differ")
    require(set(tensors) == registry_names, f"{model}: manifest/registry tensor name sets differ")

    counts = {"float16": 0, "opaque8": 0}
    raw_total = 0
    covered = bytearray(len(blob))
    finite_values = 0
    for name, t in tensors.items():
        ref = derived[name]
        off, size = t["offset"], t["size"]
        require(off % 16 == 0, f"{name}: offset {off} not 16-byte aligned")
        require(off + size <= len(blob), f"{name}: extends beyond bundle")
        require(not any(covered[off:off + size]), f"{name}: overlaps another tensor")
        covered[off:off + size] = b"\x01" * size

        payload = blob[off:off + size]
        raw = Path(ref["file"]).read_bytes()
        require(payload == raw, f"{name}: bundle bytes differ from raw parameter file")
        digest = hashlib.sha256(payload).hexdigest()
        require(digest == ref["sha256"] == t["sha256"], f"{name}: SHA-256 mismatch")
        require(t["dtype"] == ref["dtype"], f"{name}: dtype mismatch")
        counts[t["dtype"]] += 1
        raw_total += size

        if t["dtype"] == "float16":
            shape = ref["logical_shape"]
            require(shape == t["logical_shape"], f"{name}: logical shape mismatch")
            require(math.prod(shape) * 2 == size, f"{name}: size {size} != prod({shape})*2")
            values = [v for (v,) in struct.iter_unpack("<e", payload)]
            require(all(math.isfinite(v) for v in values), f"{name}: non-finite FP16 value")
            finite_values += len(values)
            if len(shape) == 4:
                k, c, r, s = shape
                require(t["physical_shape"] == [k, r, s, c], f"{name}: physical KRSC shape mismatch")
            if t["kind"] == "bias" and t["op"] == "1x1":
                if t["layer"] == "sr":
                    require(any(v != 0.0 for v in values), f"{name}: SR head 1x1 bias unexpectedly all zero")
                else:
                    require(all(b == 0 for b in payload), f"{name}: skip projection bias not exactly zero")
        else:
            require(t["logical_shape"] is None and ref["logical_shape"] is None,
                    f"{name}: opaque8 entry must have no logical shape")
            require(size == ref["bytes"] == len(raw), f"{name}: opaque8 size mismatch")

    padding = [blob[i] for i in range(len(blob)) if not covered[i]]
    require(all(b == 0 for b in padding), f"{model}: non-zero alignment padding")
    require(counts["float16"] == exp["float16"] and counts["opaque8"] == exp["opaque8"],
            f"{model}: dtype counts {counts}")
    require(raw_total == exp["raw"] == manifest["raw_payload_bytes"], f"{model}: raw total {raw_total}")
    return {"model": model, "aligned": len(blob), "raw": raw_total, "counts": counts, "values": finite_values}


def main() -> int:
    derived_doc = json.loads((RESEARCH_DIR / "all-tensors-derived.json").read_text())
    registries = json.loads((RESEARCH_DIR / "all-parameter-registries.json").read_text())
    derived = {m["model"]: {t["name"]: t for t in m["tensors"]} for m in derived_doc["models"]}
    registry_names = {r["model"]: {e["name"] for e in r["entries"]} for r in registries}
    try:
        for model in EXPECTED:
            r = verify_model(model, derived[model], registry_names[model])
            print(
                f"PASS: {r['counts']['float16'] + r['counts']['opaque8']} tensors packed for {model}.bin "
                f"({r['aligned']:,} bytes aligned, {r['raw']:,} bytes raw; "
                f"{r['counts']['float16']} float16, {r['counts']['opaque8']} opaque8)"
            )
        print("PASS: All FP16 values finite; SHA-256 digests match reference; skip biases exactly zero.")
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
