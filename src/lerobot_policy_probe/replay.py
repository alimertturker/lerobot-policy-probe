"""Replay whole episodes and record what the policy would have commanded.

`score` gives numbers. This gives the picture: for randomly chosen episodes, what the human
commanded each joint to do, against what the policy would have commanded from the same
observations.

Execution mimics deployment rather than teacher forcing. Every `replan` steps the policy
sees the real observation at that frame and predicts a chunk, and the chunk runs open loop
until the next replan, exactly as a chunked policy drives an arm. So the recorded trace is
a fair picture of what the arm would do, not a per-frame best case.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def episode_bounds(ds, ep: int) -> tuple[int, int]:
    """(first, last_exclusive) global frame index.

    lerobot 0.6.x exposes this on the metadata table rather than as an index tensor.
    """
    row = ds.meta.episodes[int(ep)]
    return int(row["dataset_from_index"]), int(row["dataset_to_index"])


def replay(
    policy, pre, post, ds, out: Path, *, label: str,
    episodes: int = 5, replan: int = 50, max_steps: int = 600, seed: int = 0,
) -> list[dict]:
    """Write one .npz of (real, model) joint traces per episode. Returns a summary."""
    import torch

    out.mkdir(parents=True, exist_ok=True)
    cams = [k.removeprefix("observation.images.") for k in ds.features if k.startswith("observation.images.")]
    joints = ds.features["action"].get("names") or [
        f"joint_{i}" for i in range(ds.features["action"]["shape"][0])
    ]

    rng = np.random.default_rng(seed)
    eps = sorted(rng.choice(ds.num_episodes, size=min(episodes, ds.num_episodes), replace=False).tolist())

    summary = []
    for ep in eps:
        lo, hi = episode_bounds(ds, ep)
        n = min(hi - lo, max_steps)
        real = np.stack([ds[lo + k]["action"].float().numpy() for k in range(n)])
        model = np.full_like(real, np.nan)

        for a in range(0, n, replan):
            item = ds[lo + a]
            obs = {"observation.state": item["observation.state"].unsqueeze(0).float()}
            for cam in cams:
                obs[f"observation.images.{cam}"] = item[f"observation.images.{cam}"].float().unsqueeze(0)
            obs["task"] = item.get("task", "")
            torch.manual_seed(seed + lo + a)
            with torch.no_grad():
                chunk = post(policy.predict_action_chunk(pre(obs)))
            pred = chunk.squeeze(0).float().cpu().numpy()
            span = min(len(pred), replan, n - a)
            model[a : a + span] = pred[:span]

        mae = float(np.nanmean(np.abs(model - real)))
        summary.append({"episode": int(ep), "steps": int(n), "mae": mae})
        np.savez(
            out / f"{label}_ep{ep:03d}.npz",
            real=real, model=model, joints=np.array(joints, dtype=object), replan=replan,
        )
        print(f"  episode {ep:>3}: {n} steps, MAE {mae:.3f}")
    return summary
