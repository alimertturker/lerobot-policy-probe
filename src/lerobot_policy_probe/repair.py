"""Repair a pi05 checkpoint whose vision tower keys predate the transformers nesting.

Transformers moved PaliGemma's SigLIP tower under `.vision_model.`. A checkpoint saved by
an older transformers stores the flat names, so on a newer one all 437 vision tensors miss
on load. `PI05Policy.from_pretrained` catches that, logs a single warning, and returns the
model anyway: everything except the vision tower keeps its trained weights.

Renaming fixes the names. It does **not** fix a checkpoint whose vision tower was random to
begin with, because the training run hit the same mismatch. Run `lerobot-probe inspect`
first: if the vision tower reads RANDOM, there is nothing to recover and the model has to
be retrained from a base that loads cleanly.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

OLD = "vision_tower."
NEW = "vision_tower.vision_model."


def resolve(src: str) -> Path:
    """Accept a local directory or a hub id already present in the local cache."""
    p = Path(src).expanduser()
    if p.is_dir():
        return p
    cache = Path.home() / ".cache/huggingface/hub" / f"models--{src.replace('/', '--')}" / "snapshots"
    snaps = sorted(cache.glob("*")) if cache.is_dir() else []
    if not snaps:
        raise SystemExit(f"Not a directory and not in the local hub cache: {src}")
    return snaps[0]


def survey(keys) -> tuple[int, int]:
    """(vision tensors, of those already nested)."""
    vis = [k for k in keys if ".vision_tower." in k]
    return len(vis), sum(1 for k in vis if ".vision_model." in k)


def migrate(src_dir: Path, dst_dir: Path) -> dict:
    """Write a repaired copy. The source is never modified."""
    from safetensors.torch import load_file, save_file

    weights = src_dir / "model.safetensors"
    if not weights.exists():
        raise SystemExit(f"No model.safetensors in {src_dir}")

    sd = load_file(str(weights))
    total, nested = survey(sd.keys())
    if total == 0:
        raise SystemExit("No vision_tower tensors; this does not look like a pi05 checkpoint.")
    if nested == total:
        return {"renamed": 0, "note": "already correct"}

    dst_dir.mkdir(parents=True, exist_ok=True)
    # Copy everything but the weights, dereferencing symlinks: the hub cache stores each
    # file as a link into ../../blobs, and a plain copy leaves those dangling.
    for item in src_dir.iterdir():
        if item.name == "model.safetensors":
            continue
        target = dst_dir / item.name
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        elif target.exists() or target.is_symlink():
            target.unlink()
        if item.is_dir():
            shutil.copytree(item, target, symlinks=False)
        else:
            shutil.copy(item, target)

    out, renamed = {}, 0
    for k, v in sd.items():
        if ".vision_tower." in k and ".vision_model." not in k:
            k = k.replace(OLD, NEW, 1)
            renamed += 1
        out[k] = v
    save_file(out, str(dst_dir / "model.safetensors"), metadata={"format": "pt"})

    # Training-only flag; the official pi05 checkpoints ship it false.
    cfg_path = dst_dir / "config.json"
    changed_cfg = False
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("gradient_checkpointing"):
            cfg["gradient_checkpointing"] = False
            cfg_path.write_text(json.dumps(cfg, indent=2))
            changed_cfg = True

    return {"renamed": renamed, "total": len(sd), "gradient_checkpointing_disabled": changed_cfg}
