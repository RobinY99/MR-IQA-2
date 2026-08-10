# Evaluation guide

Two evaluation levels are provided:

1. **Actor-only:** structured-output validity and rating PLCC/SRCC/MAE.
2. **Full:** Actor inference followed by frozen Editor and frozen E5 Judge,
   preserving solution text, edited-image provenance, and quality delta.

## Dataset sets

| Driver mode | Manifests | Rows |
| --- | --- | ---: |
| `validation` | fixed validation split | 200 |
| `test` | KonIQ, SPAQ, LIVE-W, KADID-10K, AGIQA-3K, CSIQ | 28,270 |
| `all` | validation plus all six test manifests | 28,470 |

Every source row remains in the output; invalid Actor rows receive zero-filled
reasoning reward in all-row aggregates.

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

Replace `validation` with `test` for all six datasets. Eight Actor shards merge
into one `merged.json` per dataset with metrics and retained completions.

## Full Actor→Editor→Judge evaluation

Configure the frozen Editor, source E5 Judge, their manifests, and the
original-score cache in `.env` using `.env.example`. The launcher validates
the configured artifacts automatically:

```dotenv
DIFFUSERS_MODEL_PATH=<local-black-forest-labs-FLUX.2-klein-4B-revision-e7b7dc27>
JUDGE_MODEL_PATH=<repository-root>/checkpoints/mr-iqa-2/judge/source-e5
JUDGE_MANIFEST_PATH=<repository-root>/checkpoints/mr-iqa-2/judge/source-e5/provenance.json
```

The Editor is
[`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294)
at revision `e7b7dc27f91deacad38e78976d1f2b499d76a294` (Apache-2.0). Obtain it
from the upstream repository.

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

Set `EVAL_OUTPUT_ROOT` for contracts, merged Actor files, Editor/Judge results,
edited images, telemetry, and summaries.

## Artifact validation and checkpoint promotion

The evaluator validates the Actor into `contract.json`; only a matching,
fully validated checkpoint can be promoted.

## Resume behavior

Set `EVAL_RESUME=1` to resume when the checkpoint, data, images, prompt,
Editor/Judge versions, and score-cache contract match.

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

PLCC/SRCC measure rating behavior, so audit solution diversity separately. The
completion E5 Actor reaches KonIQ PLCC/SRCC 0.931863/0.913624 while all 2,010
normalized solutions are identical. Different source images still require
per-image Editor/Judge evaluation.

## Comparing checkpoints

Use [`../scripts/compare_validation.sh`](../scripts/compare_validation.sh) with
the same 200 rows, Judge, prompt, Editor, score cache, sampling profile, and
metric denominator.
