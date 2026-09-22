"""Render the traces from `replay` as demonstrated-vs-commanded joint angle graphs.

One figure per episode, one panel per joint. Solid black is what the human commanded,
dashed is what each policy would have commanded from the same observations. Vertical lines
mark the replans, where the policy saw a fresh observation.

Reading them:

* tracks the human, drifting between replans  -> healthy; the sawtooth is normal
* offset from the first sample                -> not tracking the arm; on hardware this is
                                                 a commanded jump, a lunge, then a reversal
* dithers around the human                    -> the chunk is noisy; the arm buzzes even
                                                 with zero latency
* flat                                        -> collapsed to a constant, usually untrained
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

_COLORS = {"smolvla": "tab:orange", "pi05": "tab:green", "act": "tab:blue"}


def plot(traces: Path, out: Path, fps: float = 30.0) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")  # never open a window; these runs are usually headless
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(traces / "*_ep*.npz")))
    if not files:
        raise SystemExit(f"No traces in {traces}. Run `lerobot-probe replay` first.")

    by_ep: dict[str, dict[str, Path]] = {}
    for f in files:
        p = Path(f)
        label, ep = p.stem.rsplit("_ep", 1)
        by_ep.setdefault(ep, {})[label] = p

    written = []
    for ep, entries in sorted(by_ep.items()):
        first = np.load(next(iter(entries.values())), allow_pickle=True)
        joints = list(first["joints"])
        real = first["real"]
        replan = int(first["replan"]) if "replan" in first else 50
        n, nj = real.shape
        t = np.arange(n) / fps

        fig, axes = plt.subplots(nj, 1, figsize=(13, 2.1 * nj), sharex=True)
        axes = np.atleast_1d(axes)
        for j, ax in enumerate(axes):
            ax.plot(t, real[:, j], color="black", lw=1.6, label="demonstrated (real actuator)")
            for label, path in sorted(entries.items()):
                m = np.load(path, allow_pickle=True)["model"]
                ax.plot(t[: len(m)], m[:, j], lw=1.3, ls="--",
                        color=_COLORS.get(label), label=f"{label} (model command)")
            for b in range(replan, n, replan):
                ax.axvline(b / fps, color="grey", alpha=0.25, lw=0.7)
            ax.set_ylabel(str(joints[j]).replace(".pos", ""), fontsize=9)
            ax.grid(alpha=0.25)
            if j == 0:
                ax.legend(loc="upper right", fontsize=8, ncol=len(entries) + 1)
        axes[-1].set_xlabel("seconds (grey lines = policy replans)")
        fig.suptitle(f"Episode {ep} — demonstrated vs model-commanded joint angles", fontsize=12)
        fig.tight_layout()
        path = out / f"episode_{ep}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
        print(f"  wrote {path}")
    return written
