"""`lerobot-probe` — find out whether a policy is trained, and whether it can drive an arm.

Five subcommands, cheapest first. Run them in order; each one rules out a layer.

    inspect   are the weights actually there?           no GPU, no dataset, no download
    score     does it beat doing nothing?               needs the dataset
    replay    what would it command, minute by minute?  needs the dataset
    plot      draw that                                 needs matplotlib
    repair    fix pi05 vision key names                 only if inspect says names, not noise
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_policy(policy_type: str, checkpoint: str, device: str):
    """Policy plus its own pre/post processors, with only the device overridden.

    The checkpoint's saved pipeline carries its rename map, which is what training and
    lerobot-rollout both use. Overriding the whole preprocessor would silently drop it.
    """
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    policy = get_policy_class(policy_type).from_pretrained(checkpoint)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu", "float_dtype": "float32"}},
    )
    return policy, pre, post


def cmd_inspect(args) -> None:
    from .inspect_weights import inspect, render

    findings = inspect(args.checkpoint, args.base, per_component=args.probes)
    print(render(findings, args.checkpoint, args.base))
    if args.json:
        Path(args.json).write_text(
            json.dumps([f.__dict__ for f in findings], indent=2)
        )
        print(f"wrote {args.json}")


def cmd_score(args) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .score import score

    ds = LeRobotDataset(args.dataset)
    print(f"dataset: {ds.num_frames} frames, {ds.num_episodes} episodes")
    policy, pre, post = _load_policy(args.policy_type, args.checkpoint, args.device)
    result = score(
        policy, pre, post, ds,
        trials=args.trials, horizon=args.horizon, seed=args.seed, device=args.device,
    )
    print(result.render(args.checkpoint, args.dataset))


def cmd_replay(args) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .replay import replay

    ds = LeRobotDataset(args.dataset)
    print(f"dataset: {ds.num_frames} frames, {ds.num_episodes} episodes")
    policy, pre, post = _load_policy(args.policy_type, args.checkpoint, args.device)
    summary = replay(
        policy, pre, post, ds, Path(args.out),
        label=args.label or args.policy_type,
        episodes=args.episodes, replan=args.replan, max_steps=args.max_steps, seed=args.seed,
    )
    overall = sum(s["mae"] for s in summary) / max(len(summary), 1)
    print(f"\noverall MAE {overall:.3f}   traces in {args.out}")
    print(f"draw them:  lerobot-probe plot --traces {args.out}")


def cmd_plot(args) -> None:
    from .plot import plot

    plot(Path(args.traces), Path(args.out or args.traces), fps=args.fps)


def cmd_repair(args) -> None:
    from safetensors.torch import load_file

    from .repair import migrate, resolve, survey

    src = resolve(args.checkpoint)
    total, nested = survey(load_file(str(src / "model.safetensors")).keys())
    print(f"source : {src}")
    print(f"vision : {total} tensors, {nested} already nested")
    if nested == total:
        print("VERDICT: already correct, nothing to do.")
        return
    print(f"VERDICT: {total - nested} use the old flat layout and will NOT load.")
    if not args.out:
        print("\nPass --out to write a repaired copy. The source is never modified.")
        print("First run `lerobot-probe inspect` — if the vision tower reads RANDOM,")
        print("renaming will not help and the model must be retrained.")
        return
    info = migrate(src, Path(args.out).expanduser())
    print(f"renamed: {info['renamed']} tensors")
    print(f"written: {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="lerobot-probe", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="are the weights real, frozen, or random?")
    p.add_argument("checkpoint", help="hub id, local dir, or https URL to model.safetensors")
    p.add_argument("--base", default=None, help="what it was fine-tuned from, e.g. lerobot/pi05_base")
    p.add_argument("--probes", type=int, default=3, help="tensors sampled per component")
    p.add_argument("--json", default=None, help="also write findings here")
    p.set_defaults(func=cmd_inspect)

    for name, helptext, fn in (
        ("score", "does it beat doing nothing?", cmd_score),
        ("replay", "record commanded vs demonstrated joint angles", cmd_replay),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--policy-type", required=True, choices=["smolvla", "pi05", "act"])
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--dataset", required=True, help="the data the policy was TRAINED on")
        p.add_argument("--device", default="cuda")
        p.add_argument("--seed", type=int, default=0)
        if name == "score":
            p.add_argument("--trials", type=int, default=24)
            p.add_argument("--horizon", type=int, default=50)
        else:
            p.add_argument("--episodes", type=int, default=5)
            p.add_argument("--replan", type=int, default=50)
            p.add_argument("--max-steps", type=int, default=600)
            p.add_argument("--out", default="outputs/replay")
            p.add_argument("--label", default=None, help="name in filenames; defaults to policy type")
        p.set_defaults(func=fn)

    p = sub.add_parser("plot", help="draw the traces from replay")
    p.add_argument("--traces", default="outputs/replay")
    p.add_argument("--out", default=None)
    p.add_argument("--fps", type=float, default=30.0)
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("repair", help="fix pi05 vision key names")
    p.add_argument("checkpoint")
    p.add_argument("--out", default=None, help="destination for the repaired copy")
    p.set_defaults(func=cmd_repair)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
