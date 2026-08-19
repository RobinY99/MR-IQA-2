# Architecture and role boundaries

MR-IQA-2 separates one trainable Actor from a frozen Editor and Judge.

## Actor

The Actor is the only trainable model. For each image it emits one structured
completion:

```json
{
  "reasoning": {
    "evidence": "visible quality evidence grounded in the source image",
    "solution": "an image-edit instruction intended to improve that image"
  },
  "rating": 3.50
}
```

The plugin parses field spans, computes format/rating/reasoning/soft-overlong
rewards, and routes them according to the selected profile. The ViT and
multimodal aligner remain frozen.

## Editor

The Editor is the frozen `editor/` model from
[`RobinY99/MR-IQA-2`](https://huggingface.co/RobinY99/MR-IQA-2), copied from
`black-forest-labs/FLUX.2-klein-4B` revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`. It consumes only
`reasoning.solution`. Full evaluation completes all edits before starting the
Judge.

## Judge

The frozen source E5 Judge is deterministic under the published sampling
contract. It evaluates the original image (`J0`) and the edited image (`J1`).
Original-image scores use a validated cache during training; edited images are
judged online. The reward is:

```text
delta = J1 - J0
reasoning_raw_reward = sign(delta) * (1 - exp(-(delta^2) / 2))
```

The launcher validates the Judge, prompt, score cache, schema, and rating
interval against their manifests.

## Global orchestration

Training uses four Actor ranks and four service lanes:

```text
Actor GPU 0: 36 rows ---\
Actor GPU 1: 36 rows ----+--> merge 144 trajectories --> one optimizer update
Actor GPU 2: 36 rows ----+
Actor GPU 3: 36 rows ---/

Service GPUs 4..7: one Editor endpoint + one frozen-Judge endpoint per lane
```

Each rank draws six completions for each of six images. Per-step metrics merge
all four 36-row shards; W&B rank-0 reward keys are local diagnostics.

Offline evaluation uses all eight GPUs for Actor inference and editing,
enforces the Editor barrier, then starts the Judge.

## Credit and KL modes

`field_component_kl002` preserves field locality:

- format reward credits the formatting span;
- rating reward credits rating content;
- reasoning reward credits evidence and solution;
- soft-overlong credits the eligible completion under its length contract;
- sampled-K3 reasoning and rating component KL each use beta 0.02;
- no separate global completion KL is applied.

`completion_global_kl002` broadcasts all four scalar rewards across active,
eligible, non-padding completion tokens. Reasoning/rating component KL is off.
One sampled-K3 global completion KL with beta 0.02 is applied on the loss side,
exactly once per optimizer update. `kl_in_reward=false`: the KL is not subtracted
from trajectory rewards.

See [`../global/contracts/reward_and_kl.md`](../global/contracts/reward_and_kl.md)
for the executable contract and
[`../configs/training/mode_matrix.json`](../configs/training/mode_matrix.json)
for the machine-readable matrix.
