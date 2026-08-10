# Architecture and role boundaries

MR-IQA-2 separates the trainable policy from two frozen services. This split is
part of the experiment, not only an implementation detail.

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

The plugin parses the fields, records token spans and eligibility, computes
format/rating/reasoning/soft-overlong rewards, and routes those rewards to
tokens according to the selected training profile. The ViT and multimodal
aligner remain frozen; updates apply to the language-model part of the Actor.

## Editor

The Editor is a separately installed, frozen
[`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294)
service at revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`. It consumes
the Actor's `reasoning.solution` text only. Evidence and rating are never
appended to the edit prompt. MR-IQA-2 includes the service adapter and client
contract, but it does not include Editor training or Editor weights. Users
obtain the model independently. The pinned 4B checkpoint is Apache-2.0; its
upstream notices and model-card safety guidance still apply.

The output image is normalized back to the source dimensions when required,
and its digest, size, seed, runtime, and service status are recorded. Full
evaluation completes every edit before unloading the Editor and starting the
Judge.

## Judge

The frozen source E5 Judge is deterministic under the published sampling
contract. It evaluates the original image (`J0`) and the edited image (`J1`).
Original-image scores are served from a validated cache during training; edited
images are judged online. The reasoning delta is:

```text
delta = J1 - J0
reasoning_raw_reward = sign(delta) * (1 - exp(-(delta^2) / 2))
```

The source semantic identity, public-export integrity digest, prompt digest,
score-cache digest, schema, and accepted rating interval are checked before a
formal run. These two Judge tree values are intentionally distinct:

- `JUDGE_MODEL_TREE_SHA256=e25415173aacf515e97d5d561c6647a7a84f586061f3a9b2ab3fc079fe21be0a`
  is the source full-checkpoint semantic/cache identity carried by
  `judge/source-e5/provenance.json`;
- `JUDGE_MODEL_EXPORT_TREE_SHA256=21b232a1a30dc765f3e7cf16c00fd270e4be354615fea0120e32f975e2777e5c`
  is the integrity digest recomputed over the public ten-file Hub export.

The protocol verifies both and does not relabel the reduced public snapshot as
the source full tree. A successful Editor/Judge call does not by itself prove
semantic faithfulness: the Judge measures perceived quality change, while the
released contract contains no independent image–solution relevance Judge.

## Global orchestration

Training uses four Actor ranks and four service lanes:

```text
Actor GPU 0: 36 rows ---\
Actor GPU 1: 36 rows ----+--> merge 144 trajectories --> one optimizer update
Actor GPU 2: 36 rows ----+
Actor GPU 3: 36 rows ---/

Service GPUs 4..7: one Editor endpoint + one frozen-Judge endpoint per lane
```

Each rank draws six completions for each of six images. The learner first
computes local same-image group quantities, then follows the declared global
gather order. W&B reward keys emitted by rank 0 are local diagnostics; the
scientific per-step trajectory is the merge of all four 36-row shards.

The full offline evaluator uses all eight GPUs for Actor inference, waits for
all Actor shards, uses all eight service lanes for editing, enforces a complete
Editor barrier, and then starts the frozen Judge. This ordering prevents an
Editor and Judge from contending for the same GPU memory.

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

The two no-KL 30-step modes independently switch only the credit scope. Their
short duration makes them pipeline and routing checks, not replacements for the
formal runs.

See [`../global/contracts/reward_and_kl.md`](../global/contracts/reward_and_kl.md)
for the executable contract and
[`../configs/training/mode_matrix.json`](../configs/training/mode_matrix.json)
for the machine-readable matrix.
