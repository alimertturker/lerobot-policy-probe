"""Does the policy beat doing nothing? Offline, against real demonstrations.

A bad run on hardware has three possible causes that the run itself cannot separate: the
model, the transport, or the robot. This isolates the model. Frames from a LeRobot dataset
are replayed through the policy and the predicted action chunk is compared with what a
human actually did from that exact frame.

Three numbers, each answering a different question.

**Error against a hold-still baseline.** Repeating the current joint positions for the
whole horizon is the laziest possible policy. Anything that scores worse than that has not
learned the task, and no latency tuning will rescue it.

**Error at step 1 against step 50.** Predicting one tick ahead is close to "stay where you
are", so a healthy policy is nearly exact at step 1 and drifts over the horizon. A policy
that is already wrong at step 1 is not tracking the arm: on hardware it commands a jump,
lunges, then reverses when the next chunk corrects. That is what aggressive, rewinding
motion looks like from the outside.

**Direction reversals per joint.** How often the predicted trajectory changes direction
inside one chunk, against the demonstration. A high count means the chunk itself dithers,
so the arm buzzes even over a perfect network.

Score against the checkpoint's own training data. Other data measures generalisation,
which is a different question and will look unfairly bad.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def reversals(traj: np.ndarray) -> float:
    """Mean per-joint count of direction changes across a chunk."""
    sign = np.sign(np.diff(traj, axis=0))
    return float(((sign[1:] * sign[:-1]) < 0).sum(axis=0).mean())


@dataclass
class ScoreResult:
    hold_still: float
    policy: float
    per_step: np.ndarray
    reversals_policy: float
    reversals_demo: float
    step_policy: float
    step_demo: float
    trials: int
    horizon: int
    notes: list[str] = field(default_factory=list)

    @property
    def beats_doing_nothing(self) -> bool:
        return self.policy < self.hold_still

    def render(self, checkpoint: str, dataset: str) -> str:
        p = self.per_step
        marks = [0, min(9, p.shape[1] - 1), min(24, p.shape[1] - 1), p.shape[1] - 1]
        gain = 100 * (1 - self.policy / self.hold_still) if self.hold_still else 0.0
        lines = [
            "=" * 78,
            f"POLICY SCORE  {checkpoint}",
            f"              on {dataset}, {self.trials} frames, {self.horizon}-step horizon",
            "=" * 78,
            f"  hold-still baseline     : {self.hold_still:7.3f}",
            f"  policy error            : {self.policy:7.3f}   ({gain:+.0f}% vs doing nothing)",
            "  error at step "
            + " / ".join(str(m + 1) for m in marks)
            + "  : "
            + " / ".join(f"{p[:, m].mean():.2f}" for m in marks),
            f"  reversals per joint     : {self.reversals_policy:5.1f}   "
            f"(demonstration {self.reversals_demo:.1f})",
            f"  motion per tick         : {self.step_policy:6.3f}  "
            f"(demonstration {self.step_demo:.3f})",
            "",
        ]
        if not self.beats_doing_nothing:
            lines += [
                "  VERDICT: worse than holding still. The model is the problem, not the",
                "  network. Check the weights actually loaded: lerobot-probe inspect",
            ]
        elif p[:, 0].mean() > 0.25 * self.hold_still:
            lines += [
                "  VERDICT: already wrong on the FIRST action, so it is not tracking the arm.",
                "  Expect commanded jumps and corrective reversals on hardware.",
            ]
        else:
            lines += [
                "  VERDICT: tracks well at short horizon. If live motion is still bad, the",
                "  transport is the next suspect: measure staleness and chunk blending.",
            ]
        lines.append("=" * 78)
        return "\n".join(lines)


def score(
    policy, pre, post, ds, *, trials: int = 24, horizon: int = 50, seed: int = 0, device: str = "cuda"
) -> ScoreResult:
    """Score an already-loaded policy against an already-loaded dataset."""
    import torch

    cams = [k.removeprefix("observation.images.") for k in ds.features if k.startswith("observation.images.")]
    rng = np.random.default_rng(seed)
    idxs = sorted(rng.choice(ds.num_frames - horizon - 1, size=trials, replace=False).tolist())

    hold, gt_rev, gt_step = [], [], []
    errs, revs, steps, per_step = [], [], [], []
    for i in idxs:
        item = ds[i]
        gt = torch.stack([ds[i + k]["action"] for k in range(horizon)]).float().numpy()
        state = item["observation.state"].float().numpy()

        hold.append(np.abs(np.repeat(state[None, :], horizon, axis=0) - gt).mean())
        gt_rev.append(reversals(gt))
        gt_step.append(np.abs(np.diff(gt, axis=0)).mean())

        obs = {"observation.state": torch.from_numpy(state).unsqueeze(0).float()}
        for cam in cams:
            obs[f"observation.images.{cam}"] = item[f"observation.images.{cam}"].float().unsqueeze(0)
        obs["task"] = item.get("task", "")
        # Flow-matching policies sample noise per call, so the seed is fixed immediately
        # before each prediction or the run-to-run spread swamps the measurement.
        torch.manual_seed(seed + i)
        with torch.no_grad():
            chunk = post(policy.predict_action_chunk(pre(obs)))
        pred = chunk.squeeze(0).float().cpu().numpy()[:horizon]

        errs.append(np.abs(pred - gt).mean())
        revs.append(reversals(pred))
        steps.append(np.abs(np.diff(pred, axis=0)).mean())
        per_step.append(np.abs(pred - gt).mean(axis=1))

    return ScoreResult(
        hold_still=float(np.mean(hold)),
        policy=float(np.mean(errs)),
        per_step=np.stack(per_step),
        reversals_policy=float(np.mean(revs)),
        reversals_demo=float(np.mean(gt_rev)),
        step_policy=float(np.mean(steps)),
        step_demo=float(np.mean(gt_step)),
        trials=trials,
        horizon=horizon,
    )
