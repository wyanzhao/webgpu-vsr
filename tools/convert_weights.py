#!/usr/bin/env python3
"""
tools/convert_weights.py

Packs the reverse-engineered NVIDIA RTX Video SDK 1.1 VSR parameters for the
folded `low` and `medium` models into WebGPU-ready binary bundles.

Inputs (read-only):
  $RTX_VSR_RESEARCH_DIR/all-raw-parameters/<model>/*.bin
  $RTX_VSR_RESEARCH_DIR/all-tensors-derived.json

Outputs:
  extension/public/weights/<model>.bin            contiguous payloads, 16-byte aligned
  extension/public/weights/<model>_manifest.json  offsets, sizes, dtypes, shapes, SHA-256

FP16 weights are copied byte-for-byte. Their physical layout is KRSC
(((k*R + r)*S + s)*C + c), established for the float path in
rtx-vsr-research/semantics/HIGH-CLOSURE.md. The manifest records both the
registry's logical [K, C, R, S] extent and the physical [K, R, S, C] order the
shaders index with.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
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
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "extension" / "public" / "weights"
MODELS = ["low", "medium"]
ALIGNMENT = 16

NAME_RE = re.compile(
    r"^Flatten_DLPP_(?P<model>[a-z]+)_2x_(?P<layer>rb[0-4]|tb[0-3]|sr)_"
    r"(?P<kind>weight|bias|amax_output)_(?P<op>1x1|3x3_0|3x3_1|3x3_u|3x3)$"
)


def load_derived_metadata() -> tuple[str, dict[str, dict[str, dict]]]:
    derived_path = RESEARCH_DIR / "all-tensors-derived.json"
    if not derived_path.exists():
        raise FileNotFoundError(f"Missing derived tensor metadata: {derived_path}")
    data = json.loads(derived_path.read_text())
    meta: dict[str, dict[str, dict]] = {}
    for model in data.get("models", []):
        meta[model["model"]] = {t["name"]: t for t in model.get("tensors", [])}
    return data["source_sha256"], meta


def check_fp16_finite(name: str, payload: bytes) -> None:
    for (value,) in struct.iter_unpack("<e", payload):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite FP16 value in {name}")


def pack_model(model_name: str, tensor_meta: dict[str, dict], source_sha256: str) -> dict:
    print(f"[*] Packaging {model_name.upper()} model weights...")
    raw_dir = RESEARCH_DIR / "all-raw-parameters" / model_name
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing raw parameters directory: {raw_dir}")

    raw_files = sorted(raw_dir.glob("*.bin"))
    if not raw_files:
        raise ValueError(f"No parameter files found in {raw_dir}")

    binary = bytearray()
    raw_total = 0
    tensors: dict[str, dict] = {}

    for file_path in raw_files:
        name = file_path.stem
        info = tensor_meta.get(name)
        if info is None:
            raise KeyError(f"{name} is not registered in all-tensors-derived.json")
        match = NAME_RE.match(name)
        if match is None or match["model"] != model_name:
            raise ValueError(f"Unexpected parameter name {name}")

        payload = file_path.read_bytes()
        if len(payload) != info["bytes"]:
            raise ValueError(f"{name}: {len(payload)} bytes on disk, registry says {info['bytes']}")
        sha256 = hashlib.sha256(payload).hexdigest()
        if sha256 != info["sha256"]:
            raise ValueError(f"{name}: SHA-256 mismatch against derived registry")

        dtype = info["dtype"]
        logical = info.get("logical_shape")
        physical = None
        if dtype == "float16":
            if logical is None or math.prod(logical) * 2 != len(payload):
                raise ValueError(f"{name}: payload length does not match logical shape {logical}")
            check_fp16_finite(name, payload)
            if len(logical) == 4:
                k, c, r, s = logical
                physical = [k, r, s, c]
            else:
                physical = list(logical)

        binary.extend(b"\x00" * ((-len(binary)) % ALIGNMENT))
        offset = len(binary)
        binary.extend(payload)
        raw_total += len(payload)

        tensors[name] = {
            "offset": offset,
            "size": len(payload),
            "dtype": dtype,
            "logical_shape": logical,
            "physical_shape": physical,
            "physical_layout": "KRSC" if physical is not None and len(physical) == 4 else None,
            "layer": match["layer"],
            "kind": match["kind"],
            "op": match["op"],
            "sha256": sha256,
        }

    # Every RB/TB 1x1 skip projection is biasless (HIGH-CLOSURE.md "Biasless skip closure").
    for name, t in tensors.items():
        if t["kind"] == "bias" and t["op"] == "1x1" and t["layer"] != "sr":
            chunk = binary[t["offset"]: t["offset"] + t["size"]]
            if any(chunk):
                raise ValueError(f"{name}: 1x1 skip bias is not exactly zero")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bin_path = OUTPUT_DIR / f"{model_name}.bin"
    manifest_path = OUTPUT_DIR / f"{model_name}_manifest.json"
    bin_path.write_bytes(bytes(binary))
    manifest = {
        "model": model_name,
        "format_version": 1,
        "byte_order": "little_endian",
        "alignment_bytes": ALIGNMENT,
        "source_dll_sha256": source_sha256,
        "encoder_channels": None,
        "total_size_bytes": len(binary),
        "raw_payload_bytes": raw_total,
        "bundle_sha256": hashlib.sha256(binary).hexdigest(),
        "tensor_count": len(tensors),
        "tensors": tensors,
    }
    manifest["encoder_channels"] = [
        tensors[f"Flatten_DLPP_{model_name}_2x_rb{i}_bias_3x3_0"]["logical_shape"][0] for i in range(5)
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"[+] Wrote {bin_path.name} ({len(binary):,} bytes aligned, {raw_total:,} bytes raw, "
        f"{len(tensors)} tensors) and {manifest_path.name}"
    )
    return manifest


def main() -> int:
    source_sha256, tensor_meta = load_derived_metadata()
    for model in MODELS:
        pack_model(model, tensor_meta[model], source_sha256)
    return 0


if __name__ == "__main__":
    sys.exit(main())
