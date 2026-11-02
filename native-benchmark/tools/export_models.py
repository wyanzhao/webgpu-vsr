#!/usr/bin/env python3
"""
native-benchmark/tools/export_models.py

Prepares everything the native Metal benchmark needs to run the four reconstructed
RTX VSR float models (SDK 1.1: low, medium, high, ultra):

  build/weights/<model>.bin + <model>_manifest.json   high / ultra only (packed here)
  build/models/<model>.json                           network spec for all four models

low / medium weights are NOT repacked: their specs point at the M0 bundles in
extension/public/weights (verified against the research registry here as well).

Everything is derived from the read-only research directory:
  variant-graphs/<model>.json        node list, logical KCRS shapes, concat order, heads
  all-tensors-derived.json           byte length + SHA-256 of every raw parameter payload
  all-raw-parameters/<model>/*.bin   FP16 payloads (physical KRSC, HIGH-CLOSURE.md)
  semantics/head-folded-fixture/cbank.bin, semantics/head2-fixture/cbank.bin
                                     constructor constant banks; the stored scale-2 baseline
                                     coefficients are read from offset 0x628

The script verifies the graph has exactly the U-Net topology the Swift engine implements
(5 RBs, 4 pools, 4 TBs with nearest2x -> 3x3_u -> ELU -> concat[upsampled, skip], the head
the public SDK path selects), that every conv shape chains correctly, that every payload
matches the registry hash and contains only finite FP16 values, and that every RB/TB 1x1
skip bias is exactly zero. It never writes into the research directory.

Usage: python3 tools/export_models.py <build_dir>
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
REPO = Path(__file__).resolve().parents[2]
EXT_WEIGHTS = REPO / "extension" / "public" / "weights"
MODELS = ["low", "medium", "high", "ultra"]
PACKED_HERE = {"high", "ultra"}
ALIGNMENT = 16

# Stored scale-2 baseline coefficients (FLOAT-RECONSTRUCTION.md table). The script reads the
# authoritative values out of the constructor constant banks and refuses to continue if they
# differ from the documented table.
DOC_SCALE2 = [
    [-0.03515625, 0.26171875, 0.87890625, -0.10546875],
    [-0.10546875, 0.87890625, 0.26171875, -0.03515625],
]
CBANKS = {
    "folded": RESEARCH / "semantics" / "head-folded-fixture" / "cbank.bin",
    "unfolded": RESEARCH / "semantics" / "head2-fixture" / "cbank.bin",
}
CBANK_COEFF_OFFSET = 0x628


class SpecError(RuntimeError):
    pass


def fail(msg: str) -> None:
    raise SpecError(msg)


def half_bits_to_float(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def read_coefficients(family: str) -> list[list[float]]:
    data = CBANKS[family].read_bytes()
    coeffs = []
    for phase in range(2):
        raw = struct.unpack_from("<4H", data, CBANK_COEFF_OFFSET + 8 * phase)
        coeffs.append([half_bits_to_float(b) for b in raw])
    if coeffs != DOC_SCALE2:
        fail(f"{CBANKS[family]}: scale-2 coefficients {coeffs} differ from the documented table")
    return coeffs


def load_registry() -> tuple[str, dict[str, dict[str, dict]]]:
    data = json.loads((RESEARCH / "all-tensors-derived.json").read_text())
    reg = {m["model"]: {t["name"]: t for t in m["tensors"]} for m in data["models"]}
    return data["source_sha256"], reg


def node_map(graph: dict) -> dict[str, dict]:
    nodes = {}
    for n in graph["nodes"]:
        if n["id"] in nodes:
            fail(f"duplicate node id {n['id']}")
        nodes[n["id"]] = n
    return nodes


def expect(nodes: dict, nid: str, op: str, inputs: list[str] | None = None) -> dict:
    n = nodes.get(nid)
    if n is None:
        fail(f"missing node {nid}")
    if n["op"] != op:
        fail(f"node {nid}: op {n['op']} != {op}")
    if inputs is not None and n["inputs"] != inputs:
        fail(f"node {nid}: inputs {n['inputs']} != {inputs}")
    return n


def conv_entry(n: dict, k: int, c: int, r: int) -> dict:
    shape = n["logical_KCRS"]
    if shape != [k, c, r, r]:
        fail(f"{n['id']}: logical_KCRS {shape} != {[k, c, r, r]}")
    if n.get("weight_dtype") != "float16":
        fail(f"{n['id']}: weight dtype {n.get('weight_dtype')}")
    p = n["parameters"]
    return {"weight": p["weight"], "bias": p["bias"], "logical_KCRS": shape}


def build_spec(model: str, graph: dict) -> tuple[dict, list[str]]:
    nodes = node_map(graph)
    widths = graph["encoder_channels"]
    if len(widths) != 5:
        fail(f"{model}: expected 5 encoder widths, got {widths}")
    fold = graph["fold_factor"]
    align = graph["input_alignment"]
    expect(nodes, "preprocess", "preprocess_RGB_to_half4", [])
    if fold == 2:
        family = "folded"
        f = expect(nodes, "fold", "space_to_depth_half4", ["preprocess"])
        if f["mapping"] != "F[Y,X,8dy+4dx+c]=I[2Y+dy,2X+dx,c]":
            fail(f"{model}: unexpected fold mapping {f['mapping']}")
        first_input, in_ch = "fold", 16
    elif fold == 1:
        family = "unfolded"
        expect(nodes, "input_channel_pad", "implicit_first_kernel_half4_to_C8", ["preprocess"])
        first_input, in_ch = "input_channel_pad", 8
    else:
        fail(f"{model}: unsupported fold factor {fold}")

    convs: dict[str, dict] = {}
    prev, prev_c = first_input, in_ch
    for i in range(5):
        k = widths[i]
        convs[f"rb{i}.c0"] = conv_entry(expect(nodes, f"rb{i}.c0", "conv2d", [prev]), k, prev_c, 3)
        expect(nodes, f"rb{i}.elu0", "ELU", [f"rb{i}.c0"])
        convs[f"rb{i}.c1"] = conv_entry(expect(nodes, f"rb{i}.c1", "conv2d", [f"rb{i}.elu0"]), k, k, 3)
        expect(nodes, f"rb{i}.elu1", "ELU", [f"rb{i}.c1"])
        convs[f"rb{i}.projection"] = conv_entry(
            expect(nodes, f"rb{i}.projection", "conv2d", [prev]), k, prev_c, 1)
        expect(nodes, f"rb{i}.sum", "add", [f"rb{i}.elu1", f"rb{i}.projection"])
        if i < 4:
            p = expect(nodes, f"rb{i}.pool", "average_pool2x2", [f"rb{i}.sum"])
            if p.get("stride") != 2:
                fail(f"{model}: rb{i}.pool stride {p.get('stride')}")
            prev = f"rb{i}.pool"
        else:
            prev = f"rb{i}.sum"
        prev_c = k

    for j in range(4):
        level = 3 - j
        k = widths[level]
        expect(nodes, f"tb{j}.upsample", "nearest2x", [prev])
        convs[f"tb{j}.u"] = conv_entry(expect(nodes, f"tb{j}.u", "conv2d", [f"tb{j}.upsample"]), k, prev_c, 3)
        expect(nodes, f"tb{j}.elu_u", "ELU", [f"tb{j}.u"])
        cat = expect(nodes, f"tb{j}.concat", "concat_channels", [f"tb{j}.elu_u", f"rb{level}.sum"])
        if cat.get("order") != ["upsampled_conv_features", "encoder_skip"]:
            fail(f"{model}: tb{j}.concat order {cat.get('order')}")
        convs[f"tb{j}.c0"] = conv_entry(expect(nodes, f"tb{j}.c0", "conv2d", [f"tb{j}.concat"]), k, 2 * k, 3)
        expect(nodes, f"tb{j}.elu0", "ELU", [f"tb{j}.c0"])
        convs[f"tb{j}.c1"] = conv_entry(expect(nodes, f"tb{j}.c1", "conv2d", [f"tb{j}.elu0"]), k, k, 3)
        expect(nodes, f"tb{j}.elu1", "ELU", [f"tb{j}.c1"])
        convs[f"tb{j}.projection"] = conv_entry(
            expect(nodes, f"tb{j}.projection", "conv2d", [f"tb{j}.concat"]), k, 2 * k, 1)
        expect(nodes, f"tb{j}.sum", "add", [f"tb{j}.elu1", f"tb{j}.projection"])
        prev, prev_c = f"tb{j}.sum", k

    # Head: the public EvaluateFeature path selects the head with original-source baseline
    # scale 2 (RUNTIME-CONTRACT.md: high/ultra setter picks index int(2.0)-2 = sr_2x; folded
    # models have the single folded p4/r2 head "sr").
    heads = [h for h in graph["heads"] if h["baseline_original_scale"] == 2]
    if len(heads) != 1:
        fail(f"{model}: expected exactly one scale-2 head, found {len(heads)}")
    prefix = heads[0]["output"].rsplit(".", 1)[0]
    p = heads[0]["learned_shuffle"]
    hc0 = expect(nodes, f"{prefix}.c0", "conv2d", [prev])
    hidden = hc0["logical_KCRS"][0]
    convs["head.c0"] = conv_entry(hc0, hidden, prev_c, 3)
    expect(nodes, f"{prefix}.elu", "ELU", [f"{prefix}.c0"])
    hc1 = expect(nodes, f"{prefix}.c1", "conv2d", [f"{prefix}.elu"])
    outputs = hc1["logical_KCRS"][0]
    if outputs != 3 * p * p:
        fail(f"{model}: head outputs {outputs} != 3*{p}^2")
    convs["head.c1"] = conv_entry(hc1, outputs, hidden, 1)
    shuffle = expect(nodes, f"{prefix}.shuffle", "pixel_shuffle", [f"{prefix}.c1"])
    if shuffle["factor"] != p:
        fail(f"{model}: shuffle factor {shuffle['factor']} != {p}")
    want_phase = f"k={p * p}*color+{p}*dy+dx"
    if shuffle["phase_mapping"] != want_phase:
        fail(f"{model}: phase mapping {shuffle['phase_mapping']} != {want_phase}")
    comp = expect(nodes, f"{prefix}.compose", "bicubic_plus_bounded_residual",
                  [f"{prefix}.shuffle", "preprocess"])
    if comp["baseline_original_scale"] != 2 or comp["learned_shuffle"] != p:
        fail(f"{model}: compose node disagrees with head table")
    if comp["baseline_coordinates"] != "fx=(outputX+0.5)/2-0.5,fy=(outputY+0.5)/2-0.5":
        fail(f"{model}: baseline coordinates {comp['baseline_coordinates']}")
    if p != 2 * fold:
        fail(f"{model}: learned shuffle {p} inconsistent with fold factor {fold} at effective scale 2")

    spec = {
        "model": model,
        "family": family,
        "fold_factor": fold,
        "input_alignment": align,
        "input_channels": in_ch,
        "encoder_channels": widths,
        "head": {
            "node_prefix": prefix,
            "hidden": hidden,
            "outputs": outputs,
            "learned_shuffle": p,
            "baseline_scale": 2,
            "phase_mapping": want_phase,
            "coefficients": read_coefficients(family),
            "coefficient_source": f"{CBANKS[family]} @0x{CBANK_COEFF_OFFSET:x}",
        },
        "convs": convs,
    }
    used = []
    for c in convs.values():
        used += [c["weight"], c["bias"]]
    return spec, used


def check_payload(name: str, payload: bytes, info: dict) -> None:
    if len(payload) != info["bytes"]:
        fail(f"{name}: {len(payload)} bytes, registry says {info['bytes']}")
    if hashlib.sha256(payload).hexdigest() != info["sha256"]:
        fail(f"{name}: SHA-256 mismatch against registry")
    if info["dtype"] != "float16":
        fail(f"{name}: dtype {info['dtype']}")
    if math.prod(info["logical_shape"]) * 2 != len(payload):
        fail(f"{name}: payload length does not match logical shape {info['logical_shape']}")
    for (v,) in struct.iter_unpack("<e", payload):
        if not math.isfinite(v):
            fail(f"{name}: non-finite FP16 value")


def pack_unfolded(model: str, used: list[str], registry: dict, out_dir: Path, dll_sha: str) -> tuple[Path, Path]:
    raw_dir = RESEARCH / "all-raw-parameters" / model
    binary = bytearray()
    tensors = {}
    for name in sorted(set(used)):
        info = registry.get(name)
        if info is None:
            fail(f"{name} not in all-tensors-derived.json")
        payload = (raw_dir / f"{name}.bin").read_bytes()
        check_payload(name, payload, info)
        binary.extend(b"\x00" * ((-len(binary)) % ALIGNMENT))
        logical = info["logical_shape"]
        physical = [logical[0], logical[2], logical[3], logical[1]] if len(logical) == 4 else list(logical)
        tensors[name] = {
            "offset": len(binary),
            "size": len(payload),
            "dtype": "float16",
            "logical_shape": logical,
            "physical_shape": physical,
            "physical_layout": "KRSC" if len(logical) == 4 else None,
            "sha256": info["sha256"],
        }
        binary.extend(payload)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{model}.bin"
    man_path = out_dir / f"{model}_manifest.json"
    bin_path.write_bytes(bytes(binary))
    man_path.write_text(json.dumps({
        "model": model,
        "format_version": 1,
        "byte_order": "little_endian",
        "alignment_bytes": ALIGNMENT,
        "source_dll_sha256": dll_sha,
        "total_size_bytes": len(binary),
        "bundle_sha256": hashlib.sha256(binary).hexdigest(),
        "tensor_count": len(tensors),
        "tensors": tensors,
    }, indent=2) + "\n")
    return bin_path, man_path


def verify_extension_bundle(model: str, used: list[str], registry: dict) -> tuple[Path, Path]:
    bin_path = EXT_WEIGHTS / f"{model}.bin"
    man_path = EXT_WEIGHTS / f"{model}_manifest.json"
    if not bin_path.exists() or not man_path.exists():
        fail(f"missing M0 bundle {bin_path} / {man_path}")
    blob = bin_path.read_bytes()
    man = json.loads(man_path.read_text())
    if hashlib.sha256(blob).hexdigest() != man["bundle_sha256"]:
        fail(f"{bin_path}: bundle SHA-256 differs from manifest")
    for name in used:
        t = man["tensors"].get(name)
        if t is None:
            fail(f"{name} missing from {man_path}")
        check_payload(name, blob[t["offset"]: t["offset"] + t["size"]], registry[name])
    return bin_path, man_path


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    build = Path(sys.argv[1]).resolve()
    dll_sha, registry = load_registry()
    for model in MODELS:
        graph_path = RESEARCH / "variant-graphs" / f"{model}.json"
        graph_bytes = graph_path.read_bytes()
        spec, used = build_spec(model, json.loads(graph_bytes))
        reg = registry[model]
        for c in spec["convs"].values():
            for key in ("weight", "bias"):
                info = reg.get(c[key])
                if info is None:
                    fail(f"{c[key]} not registered")
                want = c["logical_KCRS"] if key == "weight" else [c["logical_KCRS"][0]]
                if info["logical_shape"] != want:
                    fail(f"{c[key]}: registry shape {info['logical_shape']} != graph {want}")
        if model in PACKED_HERE:
            bin_path, man_path = pack_unfolded(model, used, reg, build / "weights", dll_sha)
        else:
            bin_path, man_path = verify_extension_bundle(model, used, reg)
        # Biasless skip closure: every RB/TB 1x1 projection bias must be exactly zero.
        man = json.loads(man_path.read_text())
        blob = bin_path.read_bytes()
        for cid, c in spec["convs"].items():
            if cid.endswith(".projection"):
                t = man["tensors"][c["bias"]]
                if any(blob[t["offset"]: t["offset"] + t["size"]]):
                    fail(f"{c['bias']}: 1x1 skip bias is not exactly zero")
        spec["weights_bin"] = str(bin_path)
        spec["weights_manifest"] = str(man_path)
        spec["variant_graph"] = str(graph_path)
        spec["variant_graph_sha256"] = hashlib.sha256(graph_bytes).hexdigest()
        spec["source_dll_sha256"] = dll_sha
        (build / "models").mkdir(parents=True, exist_ok=True)
        (build / "models" / f"{model}.json").write_text(json.dumps(spec, indent=2) + "\n")
        nparams = sum(math.prod(c["logical_KCRS"]) + c["logical_KCRS"][0] for c in spec["convs"].values())
        print(f"[export] {model:6s} {spec['family']:8s} widths={spec['encoder_channels']} "
              f"head={spec['head']['node_prefix']} hidden={spec['head']['hidden']} "
              f"K={spec['head']['outputs']} p={spec['head']['learned_shuffle']} "
              f"params={nparams:,} weights={bin_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SpecError as e:
        print(f"[export] FAIL: {e}", file=sys.stderr)
        sys.exit(1)
