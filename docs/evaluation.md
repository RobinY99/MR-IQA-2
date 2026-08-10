# Evaluation guide

MR-IQA-2 provides two evaluation levels:

1. **Actor-only:** structured-output validity and rating PLCC/SRCC/MAE.
2. **Full:** Actor inference followed by frozen Editor and frozen E5 Judge,
   preserving solution text, edited-image provenance, and quality delta.

The same deterministic Actor sampling profile is used for validation and
generalization.

## Dataset sets

| Driver mode | Manifests | Rows |
| --- | --- | ---: |
| `validation` | fixed validation split | 200 |
| `test` | KonIQ, SPAQ, LIVE-W, KADID-10K, AGIQA-3K, CSIQ | 28,270 |
| `all` | validation plus all six test manifests | 28,470 |

The exact per-dataset counts are in [`data.md`](data.md). Every source row must
remain represented in the output. Invalid Actor rows receive zero-filled
reasoning reward in all-row aggregates rather than silently disappearing.

## Actor-only evaluation

Actor-only mode does not require the Editor, Judge, or original-score cache:

```bash
EVAL_ACTOR_ONLY=1 \
EVAL_NAME=field-e5-actor-only \
ACTOR_MODEL_PATH=<actor-checkpoint> \
ACTOR_PROCESSOR_PATH=<base-actor-or-processor> \
EVAL_IMAGE_ROOT=<dataset-image-root> \
bash scripts/evaluate.sh validation
```

Run all six generalization datasets by replacing `validation` with `test`.
Eight Actor shards are merged into one `merged.json` per dataset. Its `summary`
contains row coverage, format/JSON/rating validity, PLCC, SRCC, and MAE; its
`results` retain the raw completion, parsed evidence/solution/rating, token
counts, finish reason, source row, and errors.

## Full Actor→Editor→Judge evaluation

Configure the frozen Editor, source E5 Judge, their manifests, and the
original-score cache in `.env` using `.env.example`. The launcher validates
the configured artifacts automatically:

```dotenv
DIFFUSERS_MODEL_PATH=<local-black-forest-labs-FLUX.2-klein-4B-revision-e7b7dc27>
JUDGE_MODEL_PATH=<repository-root>/checkpoints/mr-iqa-2/judge/source-e5
JUDGE_MANIFEST_PATH=<repository-root>/checkpoints/mr-iqa-2/judge/source-e5/provenance.json
```

The Editor source is
[`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294)
at revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`; it is not redistributed
by MR-IQA-2. The pinned 4B Editor is Apache-2.0; retain its upstream notices
and model-card safety guidance. The launcher checks the Judge against the
downloaded artifact and manifest.

Then run:

```bash
EVAL_ACTOR_ONLY=0 \
EVAL_NAME=field-e5-full \
ACTOR_MODEL_PATH=<actor-checkpoint> \
ACTOR_PROCESSOR_PATH=<base-actor-or-processor> \
EVAL_IMAGE_ROOT=<dataset-image-root> \
bash scripts/evaluate.sh validation
```

Generalization:

```bash
EVAL_ACTOR_ONLY=0 \
EVAL_NAME=field-e5-full \
ACTOR_MODEL_PATH=<actor-checkpoint> \
EVAL_IMAGE_ROOT=<dataset-image-root> \
bash scripts/evaluate.sh test
```

The full sequence is deliberately staged:

1. use all eight GPUs for eight Actor shards;
2. merge and validate Actor outputs;
3. use eight Editor lanes and finish every edit;
4. unload Editor state at the barrier;
5. use the frozen E5 Judge for edited images;
6. combine cached original scores and edited scores into per-row deltas.

The output root includes an evaluation contract, merged Actor files, an Actor
index, Editor results, edited images, Judge results, service telemetry, and
summary JSON. Set `EVAL_OUTPUT_ROOT` to keep large generated images outside the
source tree.

## Artifact validation and checkpoint promotion

Before inference, the evaluator validates the Actor artifact automatically and
records its identity in `contract.json`. Formal per-epoch validation carries
that verified identity back to checkpoint promotion. Promotion rejects an
artifact that does not match the quarantined manifest, and the next epoch can
resolve only a fully validated, promoted checkpoint.

## Resume behavior

Set `EVAL_RESUME=1` to reuse complete merged Actor output and resume the offline
Editor/Judge state. Resume is content-aware only when the checkpoint, data,
source images, prompt, Editor/Judge versions, and score-cache provenance match.
Do not copy a partially completed state directory to a different checkpoint
and call it a resume.

## Metrics

Report Actor metrics and Editor/Judge metrics separately:

- `format` / `json_parse_success` / `rating_valid`;
- PLCC, SRCC, and MAE over valid Actor ratings;
- success, actor-ineligible, and service-error counts;
- `J0`, `J1`, `J1-J0`;
- reasoning reward over successful rows and over all rows with zero fill;
- evidence and solution exact/normalized uniqueness;
- modal solution and semantic template-family share;
- edited-image uniqueness.

PLCC/SRCC measure rating behavior. They do not test whether `solution` is
image-conditioned. The completion E5 Actor obtains KonIQ PLCC/SRCC of
0.931863/0.913624 while all 2,010 normalized solutions are the same. Always
audit fields separately.

## If solution text repeats, is Judge still necessary?

Yes when source images differ. A fixed Editor prompt applied to different
source images can produce different images and different quality changes. In
the completion E5 generalization run, all 28,270 edited images were unique even
though the solution text collapsed. Per-image Judge evaluation was therefore
still required to measure the intervention's effect.

Judge work may be reused only when the source image, normalized solution,
Editor version, seed, preprocessing contract, and output image are identical.
Even then, retain a sampled recomputation audit.
Caching identical Judge inputs saves compute; it does not repair the Actor's
loss of grounded reasoning.

## Comparing checkpoints

Use [`../scripts/compare_validation.sh`](../scripts/compare_validation.sh) for
like-for-like validation artifacts. A valid comparison must use the same 200
rows, source Judge and prompt contract, Editor contract, score cache, Actor
sampling profile, and metric denominator. Never rank checkpoints by PLCC/SRCC
alone when solution behavior is part of the task.
