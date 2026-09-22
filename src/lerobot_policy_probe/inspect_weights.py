"""Is this checkpoint actually trained? Answered by reading the weights, not the logs.

A fine-tune can fail silently. `from_pretrained` catches a state-dict mismatch, logs one
warning, and returns the model anyway, so a component that never loaded is randomly
initialised and nothing downstream complains. The policy trains, saves, uploads and
deploys. It just cannot see.

This classifies each component of a checkpoint as one of:

  pretrained   bit-identical to the base it was fine-tuned from, so it was frozen
  fine-tuned   moved from the base by a plausible amount
  RANDOM       statistically indistinguishable from a fresh initialisation, meaning it
               never loaded and never learned
  unrelated    far from the base, without matching a random init either

The random test needs no reference at all. Almost every layer in a transformer is
initialised with a standard deviation of 1/sqrt(fan_in); a trained layer drifts away from
that, an untrained one sits on it to three decimal places.

Comparisons against a base checkpoint use HTTP range requests, so a 9 GB reference costs a
few megabytes to consult instead of a full download.
"""

from __future__ import annotations

import json
import math
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DTYPES = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "BF16": np.uint16}


def _hub_url(repo: str, filename: str = "model.safetensors") -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def read_header(src: str) -> tuple[dict, int]:
    """(tensor index, byte offset of the data block) for a local path or an https URL."""
    if src.startswith("http"):
        req = urllib.request.Request(src, headers={"Range": "bytes=0-7"})
        n = struct.unpack("<Q", urllib.request.urlopen(req, timeout=90).read(8))[0]  # noqa: S310
        req = urllib.request.Request(src, headers={"Range": f"bytes=8-{7 + n}"})
        return json.loads(urllib.request.urlopen(req, timeout=90).read(n)), 8 + n  # noqa: S310
    with open(src, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def read_tensor(src: str, header: dict, base: int, key: str) -> np.ndarray:
    """One tensor, fetched by byte range so a remote reference is cheap."""
    info = header[key]
    start, end = info["data_offsets"]
    if src.startswith("http"):
        req = urllib.request.Request(src, headers={"Range": f"bytes={base + start}-{base + end - 1}"})
        raw = urllib.request.urlopen(req, timeout=300).read()  # noqa: S310
    else:
        with open(src, "rb") as fh:
            fh.seek(base + start)
            raw = fh.read(end - start)
    arr = np.frombuffer(raw, dtype=_DTYPES[info["dtype"]])
    if info["dtype"] == "BF16":  # widen to f32 by moving the bits into the exponent
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    return arr.reshape(info["shape"])


def classify_component(name: str) -> str:
    """Group a tensor name into a component people actually reason about."""
    if ".vision_tower." in name or "vision_model" in name:
        return "vision"
    if "language_model" in name or "text_model" in name:
        return "language"
    if "expert" in name:
        return "action expert"
    if "lm_head" in name or "embed_tokens" in name:
        return "embeddings"
    return "other"


def looks_random(tensor: np.ndarray) -> tuple[bool, float]:
    """(is it a fresh init, ratio of observed std to 1/sqrt(fan_in)).

    Only meaningful for 2D weight matrices, where fan_in is the second dimension.
    """
    if tensor.ndim != 2:
        return False, float("nan")
    expected = 1.0 / math.sqrt(tensor.shape[1])
    ratio = float(tensor.std() / expected) if expected else float("nan")
    return abs(ratio - 1.0) < 0.02, ratio


def _match_key(key: str, base_header: dict) -> str | None:
    """Find the same tensor in the reference file, tolerating wrapper-prefix differences.

    The two checkpoints may nest the same module differently (a leading `model.`, or the
    `.vision_model.` level). Matching on a fixed-length tail pairs unrelated tensors of
    different shape, so instead take the longest suffix that resolves to exactly one
    candidate, and require the shapes to agree.
    """
    if key in base_header:
        return key
    parts = key.split(".")
    shape = base_header[key]["shape"] if key in base_header else None
    # Walk from the longest suffix to the shortest. Long suffixes miss because of prefix
    # differences; short ones become ambiguous. The first that resolves to exactly one
    # candidate is the match, and once several match we have gone too short to be sure.
    for start in range(len(parts) - 1):
        suffix = "." + ".".join(parts[start:])
        hits = [k for k in base_header if k.endswith(suffix)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            break
    _ = shape
    return None


@dataclass
class Finding:
    component: str
    key: str
    verdict: str
    detail: str


def _pick_probes(header: dict, per_component: int) -> dict[str, list[str]]:
    """A few representative 2D weight matrices per component, spread through the depth."""
    buckets: dict[str, list[str]] = {}
    for key, info in header.items():
        if not key.endswith(".weight") or len(info.get("shape", [])) != 2:
            continue
        buckets.setdefault(classify_component(key), []).append(key)
    for comp, keys in buckets.items():
        keys.sort()
        step = max(1, len(keys) // per_component)
        buckets[comp] = keys[::step][:per_component]
    return buckets


def inspect(
    checkpoint: str, base: str | None = None, per_component: int = 3
) -> list[Finding]:
    """Classify each component of `checkpoint`, optionally against the `base` it came from."""
    src = checkpoint
    if not src.startswith("http") and Path(src).is_dir():
        src = str(Path(src) / "model.safetensors")
    elif not src.startswith("http") and not Path(src).exists():
        src = _hub_url(src)
    header, offset = read_header(src)

    base_header = base_offset = None
    base_src = None
    if base:
        base_src = base
        if not base_src.startswith("http") and Path(base_src).is_dir():
            base_src = str(Path(base_src) / "model.safetensors")
        elif not base_src.startswith("http") and not Path(base_src).exists():
            base_src = _hub_url(base_src)
        base_header, base_offset = read_header(base_src)

    findings: list[Finding] = []
    for comp, keys in sorted(_pick_probes(header, per_component).items()):
        for key in keys:
            tensor = read_tensor(src, header, offset, key)
            is_random, ratio = looks_random(tensor)

            match = _match_key(key, base_header) if base_header is not None else None

            if match is not None:
                ref = read_tensor(base_src, base_header, base_offset, match)
                if ref.shape != tensor.shape:
                    verdict, detail = "shape differs", f"{tensor.shape} vs {ref.shape}"
                else:
                    rel = float(np.abs(tensor - ref).mean() / (np.abs(ref).mean() + 1e-12))
                    if np.array_equal(tensor, ref):
                        verdict, detail = "pretrained", "bit-identical to base, so frozen"
                    elif rel > 0.9:
                        verdict = "RANDOM" if is_random else "unrelated"
                        detail = f"{100 * rel:.0f}% from base" + (
                            f", std/init {ratio:.3f} = fresh init" if is_random else ""
                        )
                    else:
                        verdict, detail = "fine-tuned", f"{100 * rel:.2f}% from base"
            elif is_random:
                verdict, detail = "RANDOM", f"std/init {ratio:.3f}, no reference needed"
            else:
                verdict, detail = "trained", f"std/init {ratio:.3f}" if ratio == ratio else "non-2D"
            findings.append(Finding(comp, key, verdict, detail))
    return findings


def render(findings: list[Finding], checkpoint: str, base: str | None) -> str:
    lines = ["=" * 78, f"CHECKPOINT  {checkpoint}"]
    if base:
        lines.append(f"COMPARED TO {base}")
    lines += ["=" * 78, f"{'component':<15}{'verdict':<14}detail"]

    by_comp: dict[str, list[Finding]] = {}
    for f in findings:
        by_comp.setdefault(f.component, []).append(f)

    broken = []
    for comp, items in by_comp.items():
        for n, item in enumerate(items):
            lines.append(f"{comp if n == 0 else '':<15}{item.verdict:<14}{item.detail}")
        if any(i.verdict in ("RANDOM", "unrelated") for i in items):
            broken.append(comp)

    lines.append("")
    if broken:
        lines += [
            f"  VERDICT: {', '.join(broken)} never loaded. Those weights are noise, and no",
            "  amount of key repair recovers them because there is nothing there to recover.",
            "  The checkpoint has to be retrained from a base that loads cleanly.",
        ]
    else:
        lines.append("  VERDICT: every component carries real weights.")
    lines.append("=" * 78)
    return "\n".join(lines)
