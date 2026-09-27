# lerobot-policy-probe

Find out whether a LeRobot policy is actually trained, and whether it can drive an arm,
before you put it on hardware.

> **Status: early, personal-project quality.** Built while debugging one arm and two
> checkpoints. The numbers below are from that setup and are not a general claim.

## Why this exists

A fine-tune can fail silently. `from_pretrained` catches a state-dict mismatch, logs one
warning line, and returns the model anyway. A component that never loaded stays randomly
initialised, and nothing downstream complains. The policy trains, saves, uploads, deploys,
and drives your arm. It is simply blind.

That is not hypothetical. It is what these tools were written to find:

```
component      verdict       detail
action expert  fine-tuned    0.85% from base
language       pretrained    bit-identical to base, so frozen
vision         RANDOM        137% from base, std/init 1.000 = fresh init
```

That checkpoint had been fine-tuned, uploaded, and deployed. Its vision encoder was noise
the whole time. On the arm it looked like a latency problem: aggressive motion that
reversed on itself. Weeks of tuning the network would not have touched it.

## The ladder

Run these in order. Each rules out a layer, cheapest first.

| step | question | needs |
|---|---|---|
| `inspect` | are the weights actually there? | nothing. No GPU, no dataset, no download |
| `score` | does it beat doing nothing? | the dataset |
| `replay` + `plot` | what would it command, second by second? | the dataset, matplotlib |
| `repair` | fix pi05 vision key names | only when `inspect` says names, not noise |

If `inspect` or `score` fails, stop. The model is the problem and no amount of deployment
tuning will help.

## Install

```bash
pip install "lerobot @ git+https://github.com/huggingface/lerobot.git"   # 0.6.2 is not on PyPI yet
pip install -e .                    # lerobot>=0.6.2 must already be importable
pip install -e ".[dataset,plot]"    # score, replay and plot need these
```

## Running in Google Colab

Useful when you have no local GPU. `inspect` and `repair` run on the free CPU runtime;
`score` and `replay` load the whole policy and want a GPU (**Runtime → Change runtime type
→ T4 GPU** or better; pi05 needs an A100 or L4).

```python
# 1. install. lerobot>=0.6.2 is not on PyPI yet, so it comes from GitHub, first.
!pip install -q "lerobot[dataset,smolvla,pi] @ git+https://github.com/huggingface/lerobot.git"
!pip install -q "lerobot-policy-probe[plot] @ git+https://github.com/alimertturker/lerobot-policy-probe.git"
```

```python
# 2. private checkpoints or datasets: add HF_TOKEN under the key icon (Secrets) in the
#    left sidebar and allow notebook access. huggingface_hub picks it up automatically.
#    Or log in interactively:
from huggingface_hub import login
login()
```

```python
# 3. run the ladder
!lerobot-probe inspect lerobot/smolvla_base

!lerobot-probe score --policy-type smolvla \
  --checkpoint alimerido/smolvla-wrist-top-cube-v3 \
  --dataset alimerido/wrist-top-cube_20260705_134536 --device cuda

!lerobot-probe replay --policy-type smolvla \
  --checkpoint alimerido/smolvla-wrist-top-cube-v3 \
  --dataset alimerido/wrist-top-cube_20260705_134536 --episodes 3 --out outputs/replay
!lerobot-probe plot --traces outputs/replay
```

```python
# 4. show the plots inline
from pathlib import Path
from IPython.display import Image, display
for png in sorted(Path("outputs/replay").glob("*.png")):
    display(Image(str(png)))
```

For a complete worked example that clones, installs, and replays smolvla and pi05 on the same
dataset, see [`examples/replay_colab.ipynb`](examples/replay_colab.ipynb).

Colab notes:

- Python must be 3.12 or newer, which current Colab runtimes are. Check with `!python --version`.
- If pip reports a conflict with Colab's preinstalled torch, restart the session
  (**Runtime → Restart session**) after installing and run from step 2.
- `inspect` reads remote checkpoints over plain HTTP and does not send your token, so it only
  works on public hub repos. For a private one, download it first and pass the directory:
  `!hf download <repo> --local-dir ckpt && lerobot-probe inspect ckpt`.
- The Colab disk is wiped when the runtime stops. Copy anything you want to keep to Drive
  (`from google.colab import drive; drive.mount("/content/drive")`) or download it from the
  file browser.
- `--device cpu` works for `score` on small policies such as ACT, but is slow.

## 1. inspect — are the weights real?

```bash
lerobot-probe inspect alimerido/my-pi05 --base lerobot/pi05_base
```

Classifies each component as **pretrained** (bit-identical to the base, so frozen),
**fine-tuned** (moved by a plausible amount), **RANDOM** (indistinguishable from a fresh
initialisation), or **unrelated**.

Two things make this cheap. The random test needs no reference at all: almost every
transformer layer is initialised with a standard deviation of `1/sqrt(fan_in)`, and a
trained layer drifts off that value while an untrained one sits on it to three decimals. And
the comparison against a base uses HTTP range requests, so consulting a 9 GB reference
costs a few megabytes rather than a full download.

`--base` is optional. Without it you still get the random test, which is the one that
catches silent load failures.

## 2. score — does it beat doing nothing?

```bash
lerobot-probe score --policy-type smolvla \
  --checkpoint alimerido/smolvla-wrist-top-cube-v3 \
  --dataset alimerido/wrist-top-cube_20260705_134536 --device cuda
```

Replays real frames through the policy and compares the predicted chunk against what a
human did from that exact frame. Three numbers, each answering a different question:

| number | meaning |
|---|---|
| error against a hold-still baseline | repeating the current joint positions is the laziest possible policy. Scoring worse than it means the model has not learned the task |
| error at step 1 against step 50 | predicting one tick ahead is close to "stay where you are". A healthy policy is near-exact at step 1 and drifts. Wrong at step 1 means it is not tracking the arm, so on hardware it commands a jump, lunges, then reverses when the next chunk corrects |
| direction reversals per joint | how often the predicted trajectory changes direction inside one chunk, against the demonstration. High means the chunk itself dithers, so the arm buzzes even with zero latency |

Measured on one arm, 24 frames, hold-still baseline 9.118 and demonstrations reversing 0.4
times per joint:

| | error | vs doing nothing | step 1 | reversals |
|---|---|---|---|---|
| smolvla | 8.208 | +10% | 3.05 | 23.9 |
| pi05, vision random | 10.538 | **-16%** | **11.59** | 25.9 |

**Score against the checkpoint's own training data.** Other data measures generalisation,
which is a different question and will look unfairly bad. You can verify which dataset a
checkpoint was trained on by comparing its baked normalizer statistics against the
dataset's own.

## 3. replay and plot — what would it actually command?

```bash
lerobot-probe replay --policy-type smolvla --checkpoint <ckpt> \
  --dataset <its training set> --episodes 5 --out outputs/replay
lerobot-probe plot --traces outputs/replay
```

`--episodes N` picks N episodes at random from `--seed`; `--episode-ids 0 1 2 3 4` replays
exactly those. Use the same ids for every policy you want on one figure.

One figure per episode, one panel per joint. Solid black is the human, dashed is the
policy. Point several policies at the same output directory and they land on one figure.

Execution mimics deployment rather than teacher forcing: the policy sees a real observation
every `--replan` steps and its chunk runs open loop until the next one, so the trace is what
the arm would really do.

| what you see | what it means |
|---|---|
| tracks the human, drifting between replans | healthy; the sawtooth at replans is normal |
| offset from the first sample | not tracking. Commanded jump, lunge, reversal |
| dithers around the human | noisy chunk; the arm buzzes regardless of latency |
| flat | collapsed to a constant, usually untrained |

## 4. repair — pi05 vision key names

Transformers moved PaliGemma's SigLIP tower under `.vision_model.`. A checkpoint saved by
an older transformers stores the flat names, so all 437 vision tensors miss on load.

```bash
lerobot-probe repair alimerido/my-pi05               # diagnose only
lerobot-probe repair alimerido/my-pi05 --out ~/fixed # write a repaired copy
```

The source is never modified. Renaming is all it does: the tensor values, dtypes and shapes
are untouched, and every other file is copied byte for byte.

**Run `inspect` first.** Repair fixes names. It cannot fix a checkpoint whose vision tower
was random to begin with, because the training run hit the same mismatch and there is
nothing in the file to recover. That checkpoint has to be retrained from a base that loads
cleanly, which you can confirm before spending GPU hours:

```python
model = PI05Policy.from_pretrained(BASE)          # watch for "Could not load state dict"
sd = load_file(BASE + "/model.safetensors")
k = next(k for k in sd if "vision_tower" in k and "layers.0.self_attn.q_proj.weight" in k)
assert torch.allclose(sd[k], dict(model.named_parameters())[k])
```

## Architecture

| file | responsibility |
|---|---|
| `inspect_weights.py` | safetensors range reads, random-init detection, base comparison |
| `score.py` | offline error against demonstrations, baselines, reversal counting |
| `replay.py` | episode replay with deployment-shaped chunked execution |
| `plot.py` | the graphs |
| `repair.py` | pi05 vision key migration |
| `cli.py` | `lerobot-probe` |

## Known limits

- The random-init test assumes `std = 1/sqrt(fan_in)`, which holds for most linear layers
  but not for embeddings, norms, or anything with a custom initialiser. It only examines 2D
  weight matrices.
- `--base` matching pairs tensors by the longest unambiguous name suffix. Architectures that
  reuse layer names across differently shaped modules can still mispair; shapes are reported
  when they disagree so you can see it happen.
- `score` and `replay` load the whole policy, so they need enough memory for it.
- Nothing here tests the transport. If the model passes every step and the arm still
  misbehaves, the next suspects are observation staleness and chunk blending.

## License

Apache-2.0.
