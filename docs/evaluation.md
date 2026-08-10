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

Configure the frozen Editor, source E5 Judge, their manifests/digests, and the
original-score cache in `.env`:

```dotenv
DIFFUSERS_MODEL_PATH=<local-black-forest-labs-FLUX.2-klein-4B-revision-e7b7dc27>
JUDGE_MODEL_PATH=<repository-root>/checkpoints/mr-iqa-2/judge/source-e5
JUDGE_MANIFEST_PATH=<repository-root>/checkpoints/mr-iqa-2/judge/source-e5/provenance.json
JUDGE_MODEL_TREE_SHA256=e25415173aacf515e97d5d561c6647a7a84f586061f3a9b2ab3fc079fe21be0a
JUDGE_MODEL_EXPORT_TREE_SHA256=21b232a1a30dc765f3e7cf16c00fd270e4be354615fea0120e32f975e2777e5c
```

The Editor source is
[`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294)
at revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`; it is not redistributed
by MR-IQA-2. The pinned 4B Editor is Apache-2.0; retain its upstream notices
and model-card safety guidance. The Judge source digest is the semantic/cache
identity in `provenance.json`, while the export digest verifies the public
ten-file directory. Both are required.

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

## Checkpoint digest contract

Before inference, the evaluator independently hashes the ten allowlisted
inference-export files under `ACTOR_MODEL_PATH`. If
`ACTOR_MODEL_EXPORT_TREE_SHA256` is supplied, it must match that recomputed
value. The evaluation root's `contract.json`
(`mr_iqa_evaluation_contract_v1`) records:

```json
{
  "checkpoint_digest": {
    "semantics": "selected_inference_export",
    "algorithm": "sha256(sorted(relative_path + NUL + decimal_size + NUL + file_sha256 + LF)) over the 10 allowlisted inference-export files",
    "sha256": "<recomputed-selected-export-digest>",
    "file_count": 10
  }
}
```

`semantics` defines what is identified; `algorithm` makes the canonical byte
sequence explicit; `sha256` binds the run to the actual Actor export; and
`file_count` prevents a partial allowlist from masquerading as a complete
model. Optimizer, RNG, cache, trainer, and temporary files are not inputs. This
digest is consequently different in scope from a source full-checkpoint tree
hash.

For formal per-epoch validation, the generated
`state/checkpoints/epochN.validation.json` carries these digest semantics back
to checkpoint promotion. Promotion rejects a digest, semantics, or algorithm
that does not match the quarantined manifest and re-hashes the checkpoint
before marking it usable.

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
- edited-image digest uniqueness.

PLCC/SRCC measure rating behavior. They do not test whether `solution` is
image-conditioned. The completion E5 Actor obtains KonIQ PLCC/SRCC of
0.931863/0.913624 while all 2,010 normalized solutions are the same. Always
audit fields separately.

## If solution text repeats, is Judge still necessary?

Yes when source images differ. A fixed Editor prompt applied to different
source images can produce different images and different quality changes. In
the released completion E4 and E5 generalization runs, all 28,270 edited-image
digests were unique even though the solution text collapsed. Per-image Judge
evaluation was therefore still required to measure the intervention's effect.

Judge work may be content-addressed only when the source-image digest,
normalized solution, Editor version, seed, preprocessing contract, and output
image digest are identical. Even then, retain a sampled recomputation audit.
Caching identical Judge inputs saves compute; it does not repair the Actor's
loss of grounded reasoning.

## Comparing checkpoints

Use [`../scripts/compare_validation.sh`](../scripts/compare_validation.sh) for
like-for-like validation artifacts. A valid comparison must use the same 200
rows, source Judge tree and prompt digest, Editor contract, score cache, Actor
sampling profile, and metric denominator. Never rank checkpoints by PLCC/SRCC
alone when solution behavior is part of the task.
