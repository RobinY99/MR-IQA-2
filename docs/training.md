# Training guide

This guide reproduces the released Actor training modes. The Editor and source
E5 Judge are frozen services and are never trained by these commands.

## 1. Hardware and software

The formal topology requires Linux, a CUDA 13.0-compatible driver, and eight
visible NVIDIA GPUs:

- GPUs 0–3: four Actor ranks;
- GPUs 4–7: four Editor/Judge service lanes;
- 144 trajectories per optimizer update;
- 291 optimizer updates per formal epoch;
- five epochs, ending at global step 1,455.

Create the two isolated Python 3.12.13 environments in
[`../environment/README.md`](../environment/README.md). Actor/Judge and Editor
dependencies intentionally use different pinned versions and must not be
merged into one environment.

## 2. Required artifacts

Obtain the following before training:

1. the Qwen3.5-4B-compatible initial Actor;
2. the frozen source E5 Judge and its manifest;
3. the frozen `black-forest-labs/FLUX.2-klein-4B` Editor at the pinned
   revision below;
4. source images corresponding to `data/train.jsonl`;
5. the validated original-image Judge score cache;
6. a compatible prebuilt FlashAttention wheel.

Weights published for this project are indexed at
[huggingface.co/RobinY99/MR-IQA-2](https://huggingface.co/RobinY99/MR-IQA-2).
Use an immutable revision. The launcher and preflight validate published
artifacts automatically. The base Actor, frozen Editor, source images, and
runtime wheel are not GitHub source artifacts and may have separate
distribution terms.

### Initial Actor and frozen Editor

The initial Actor is the official Apache-2.0
[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)
at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`:

```bash
python -m pip install -r requirements/publish.txt
huggingface-cli download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir checkpoints/qwen3.5-4b
```

Pin the revision in every archival run and keep the generated runtime manifest
with the experiment.

The Editor is
[`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294)
at revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`. It is not redistributed
by MR-IQA-2. Obtain it from that official source before setting
`DIFFUSERS_MODEL_PATH`. The pinned 4B model is Apache-2.0; retain the upstream
license, notices, and model-card safety guidance. The GitHub MIT License
applies only to original MR-IQA-2 code.

### Portable original-image J0 cache

The public cache is stored at
`training_assets/original_score_cache.sqlite` in the Hugging Face repository.
Download it with the pinned publishing client:

```bash
python -m pip install -r requirements/publish.txt
huggingface-cli download RobinY99/MR-IQA-2 \
  training_assets/original_score_cache.sqlite \
  --local-dir checkpoints/mr-iqa-2
```

Published cache contract:

| Property | Value |
| --- | --- |
| Relative Hub path | `training_assets/original_score_cache.sqlite` |
| Bytes | 15,003,648 |
| Rows / samples | 10,073 / 10,073 |
| Actor ID | `source-e5-judge-step725-original-score` |
| Payload schema | `vf_original_score_cache_e5_judge_e5prompt_portable_v1` |
| Observed J0 min / max / mean | 0.83 / 4.23 / 3.1357688871239398 |

The cache is a sanitized lookup artifact, not a copy of the private experiment
database. It omits absolute filesystem paths, ground-truth scores, image bytes,
raw Judge completions, and Judge reasoning evidence/solution. It retains only
the portable metadata required to validate a J0 lookup against the frozen
source E5 Judge contract. The preflight checks the downloaded cache before use.

## 3. Private machine configuration

```bash
cp .env.example .env
```

Fill `.env` with local paths and the artifact-identity fields provided by the
published manifests. At minimum, the launcher requires:

- Conda initialization and Actor/Judge Python;
- initial Actor path and manifest identity;
- training and evaluation image roots;
- Editor environment and model path;
- Judge model, manifest, prompt schema, and identity fields;
- original-score cache path, row/sample contract, schema, and identity field;
- FlashAttention wheel path and artifact identity.

Never commit `.env`. Do not put tokens in a training profile; profiles under
`configs/training/` are public scientific configuration only.

For a Judge downloaded from `judge/source-e5`, configure its local path and
provenance manifest. Use `.env.example` for the remaining preflight fields:

```dotenv
JUDGE_MODEL_PATH=<repository-root>/checkpoints/mr-iqa-2/judge/source-e5
JUDGE_MANIFEST_PATH=<repository-root>/checkpoints/mr-iqa-2/judge/source-e5/provenance.json
```

The launcher validates the configured artifact automatically before loading
the Judge.

For the portable J0 cache, the corresponding `.env` block is:

```dotenv
ORIGINAL_SCORE_CACHE_PATH=<repository-root>/checkpoints/mr-iqa-2/training_assets/original_score_cache.sqlite
ORIGINAL_SCORE_CACHE_EXPECTED_ROW_COUNT=10073
ORIGINAL_SCORE_CACHE_EXPECTED_SAMPLE_COUNT=10073
ORIGINAL_SCORE_CACHE_EXPECTED_ACTOR_IDS=source-e5-judge-step725-original-score
ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA=vf_original_score_cache_e5_judge_e5prompt_portable_v1
ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MIN=0.0
ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MAX=5.0
```

`0.0/5.0` is the accepted Judge rating interval used by cache validation. The
actual cached values occupy `0.83/4.23`, as reported above.

## 4. Inspect and validate a mode

List the exact contract without loading models:

```bash
bash scripts/train.sh --mode field_component_kl002 --print-plan
bash scripts/train.sh --mode completion_global_kl002 --print-plan
```

Validate paths, artifacts, score-cache contract, mode invariants, and service
topology:

```bash
bash scripts/train.sh --mode field_component_kl002 --validate-config
```

Validation rejects a changed data row count, missing or changed model/runtime
artifact, unfrozen ViT/aligner, or an inconsistent credit/KL combination.

## 5. One-update smoke test

Run the full Actor→Editor→Judge path for one optimizer update before a long
job:

```bash
export RUN_ID="smoke-field-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}"
export OUTPUT_ROOT="<large-storage-root>/mr-iqa-2/${RUN_ID}"
export VF_STORAGE_ROOT="${OUTPUT_ROOT}"
export VF_MIN_FREE_GIB="<host-appropriate-smoke-threshold>"
bash scripts/train.sh --mode field_component_kl002 --smoke
```

The smoke run is an infrastructure check only. It must not be compared with a
formal checkpoint. The smoke path disables W&B internally. Choose its
host-specific `VF_MIN_FREE_GIB` only after checking the filesystem that backs
the new output directory.

## 6. Formal training

The recommended mode is field-local credit with reasoning/rating component KL:

```bash
export RUN_ID="field-formal-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}"
export OUTPUT_ROOT="<large-storage-root>/mr-iqa-2/${RUN_ID}"
export VF_STORAGE_ROOT="${OUTPUT_ROOT}"
export VF_MIN_FREE_GIB=500
bash scripts/train.sh --mode field_component_kl002
```

The diagnostic completion-wide/global-KL ablation is:

```bash
export RUN_ID="completion-formal-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}"
export OUTPUT_ROOT="<large-storage-root>/mr-iqa-2/${RUN_ID}"
export VF_STORAGE_ROOT="${OUTPUT_ROOT}"
export VF_MIN_FREE_GIB=500
bash scripts/train.sh --mode completion_global_kl002
```

Every invocation must use a unique `RUN_ID` and a new output directory.
`OUTPUT_ROOT` and `VF_STORAGE_ROOT` must resolve to the same sufficiently large
filesystem; never point either variable at a previous run. The formal
five-epoch workflow recommends `VF_MIN_FREE_GIB=500`. The threshold is
configurable so a one-update smoke can reflect host capacity, but lowering it
does not reduce the space a run may actually consume.

`WANDB_MODE=offline` is supported and is the reproducible default. Set
`WANDB_MODE=online` only for an intentionally remote-tracked formal run with
credentials supplied outside the repository. As noted above, smoke disables
W&B internally regardless of the outer setting.

The formal launcher:

1. starts four frozen Editor/Judge service lanes;
2. trains exactly 291 updates;
3. discovers the single expected `checkpoint-<end-step>` and creates its
   manifest in `quarantined` state;
4. verifies the trainer exit, checkpoint artifacts, four-rank trajectory
   evidence, provenance, tracking URI, and inference identity before
   transitioning it to `technically_valid`;
5. stops the training services and performs eight-shard Actor inference on all
   200 validation rows;
6. completes every Editor record, passes the global Editor barrier, then runs
   the frozen Judge over all 200 retained rows;
7. applies the `comparison_observational` gate and transitions the manifest to
   `promoted` only when the full evaluation contract passes;
8. resolves and revalidates that promoted manifest before permitting it to seed
   the next epoch.

Observational promotion is a structural and provenance gate, not a hidden
PLCC/SRCC selection rule. It requires 200 source rows, eight Actor shards, the
published Actor schema, no missing/bad gold rows or generation exceptions,
complete 200-row Editor and Judge records, zero Editor/Judge service errors,
and proof that all edits finished before the first Judge request. The recorded
quality and collapse metrics remain observations used for reporting and model
selection.

### Checkpoint promotion records

Each epoch writes three related records below `OUTPUT_ROOT`:

| Path | Purpose |
| --- | --- |
| `state/checkpoints/epochN.json` | `vf_checkpoint_manifest_v2`; owns `quarantined → technically_valid → promoted`, parent provenance, technical evidence, observational validation, approval, and the stable checkpoint identity |
| `state/checkpoints/epochN.validation.json` | Flattened evidence from the complete 200-row Actor→Editor barrier→Judge run, including the validated checkpoint identity used by promotion |
| `state/epochN.json` | `mr_iqa_2_epoch_chain_v2`; compact epoch-chain record with steps, checkpoint/manifest paths, final status, validation paths, and checkpoint identity |

The launcher owns the checkpoint identity checks used by validation and
promotion. Optimizer state, RNG state, caches, logs, and temporary files may be
needed for full-state resumption, but they are not accepted as substitutes for
a promoted inference artifact.

Use `--epochs N` for a deliberate prefix of one to five epochs. Use
`--skip-validation` only with `--epochs 1`; the launcher rejects every other
combination. This debugging path stops after technical validation, leaves the
checkpoint `technically_valid` and `usable=false`, writes no observational
promotion, and cannot seed another epoch.

## 7. Thirty-step ablations

The two independent no-KL probes both start from the native Actor:

```bash
bash scripts/train.sh --mode field_nokl_30step
bash scripts/train.sh --mode completion_nokl_30step
```

Do not initialize the second arm from the first arm's checkpoint. Their purpose
is to isolate credit-mask plumbing over a short horizon.

## 8. Required runtime audits

A formal update is usable only when all of the following hold:

- four trajectory shards exist and each contains 36 rows;
- the merged step contains exactly 144 rows;
- loss, gradient norm, rewards, completion lengths, and KL statistics are
  finite;
- optimizer and scheduler each update once;
- no OOM, CUDA, NCCL, traceback, or runtime failure is present;
- component/global KL application counts match the selected mode.

For `completion_global_kl002`, the expected counts are:

```text
global_completion_kl_apply_count = 1
component_kl_apply_count = 0
```

For `field_component_kl002`, reasoning and rating component KL are active and
the separate global completion KL is disabled.

Always aggregate training reward from all four trajectory ranks. A W&B
`vf/*reward_mean` value is a rank-0 local value and is not the 144-row global
mean.

## 9. Monitoring collapse

Validation must retain all 200 source rows, including actor-ineligible and
service-error rows. In addition to PLCC/SRCC/MAE, monitor:

- exact and normalized solution uniqueness;
- modal normalized-solution share;
- semantic template-family share;
- evidence uniqueness;
- success, actor-ineligible, and service-error counts;
- `J0`, `J1`, delta, and zero-filled reasoning reward.

The completion-global release first crossed 50% semantic house-template share
at global step 236 and 90% at step 265. Its E1 checkpoint already produced a
house-family solution for all 197 eligible validation rows. Rating correlation
did not expose this failure. See [`checkpoints.md`](checkpoints.md).

## 10. Outputs and resumption

The public launcher writes each experiment below its selected `OUTPUT_ROOT`:

```text
runs/epoch1 ... runs/epoch5   training stages and checkpoints
evaluation/                   per-epoch validation artifacts
state/checkpoints/epochN.json checkpoint promotion manifest
state/checkpoints/epochN.validation.json 200-row observational evidence
state/epochN.json             mr_iqa_2_epoch_chain_v2 chain record
logs/                         launcher and service logs
wandb/                        local W&B files when enabled
```

Keep every checkpoint manifest, epoch state, merged validation output, and
audit record even when large checkpoint directories are later pruned. A
resumed run must resolve and validate the promoted parent manifest and preserve
the same mode, data, source Judge, prompt, and initial/reference-model
provenance.
