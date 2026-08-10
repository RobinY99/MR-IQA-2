# Reproducibility contract

MR-IQA-2 follows the public MR-IQA release pattern—pinned environments,
relative manifests, explicit launchers, validation history, and model-tree
digests—and extends it with service, reward-mask, KL-mask, and checkpoint-chain
contracts.

## Fixed formal configuration

| Property | Published value |
| --- | --- |
| Initial Actor repository | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a), Apache-2.0 |
| Initial Actor Hub revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Prepared initial Actor experiment tree SHA-256 | `eb67981666c5fc926c829b0d2b49b180c46dc6a74c1c8df895834a54c0fe33f3` |
| Frozen Editor repository | [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294) |
| Frozen Editor Hub revision | `e7b7dc27f91deacad38e78976d1f2b499d76a294` |
| Frozen Editor distribution/license | Not redistributed by MR-IQA-2; pinned 4B checkpoint is Apache-2.0 and obtained independently |
| Source E5 Judge full promotion tree SHA-256 | `e25415173aacf515e97d5d561c6647a7a84f586061f3a9b2ab3fc079fe21be0a` |
| Public E5 Judge 10-file export-tree SHA-256 | `21b232a1a30dc765f3e7cf16c00fd270e4be354615fea0120e32f975e2777e5c` |
| Judge provenance manifest | `judge/source-e5/provenance.json` |
| Judge prompt SHA-256 | `fa78a4ccfd2194a2026ff0b6b722bf22b28f8fa060389c57c4adb1618ac280f6` |
| Actor world size | 4 |
| Service GPUs | 4 |
| Generations per image | 6 |
| Images per Actor rank | 6 |
| Rows per rank / global update | 36 / 144 |
| Optimizer updates | 291 per epoch, 1,455 total |
| Learning rate | `1e-6` |
| Maximum completion length | 160 tokens |
| Soft-overlong cache | last 16 tokens before the 160-token maximum |
| ViT / aligner | frozen / frozen |
| Actor validation sampling | deterministic, seed 42 |
| Judge sampling | deterministic, temperature 0 |
| W&B formal default | `WANDB_MODE=offline`; online is opt-in; smoke disables W&B internally |

Mode-specific values live in versioned files under `configs/training/`; do not
copy them into an undocumented shell command. `scripts/train.sh --print-plan`
is the canonical human-readable resolution of a profile.

## Validation ladder

Run checks from least expensive to most expensive:

### 1. Dependency-free source release check

```bash
bash scripts/test_release.sh --static
```

This checks required files, Python and shell syntax, JSONL schemas and hashes,
unsafe symlinks, accidental model blobs, release-size policy, common private
paths, and credential patterns.

### 2. CPU contract tests

Create the lightweight test environment described in
[`../environment/README.md`](../environment/README.md), then run:

```bash
bash scripts/test_release.sh
```

These tests exercise parsing, field spans, credit masks, reward functions,
component/global KL contracts, source-score cache behavior, service routing,
and offline-evaluation eligibility without loading full model weights.

### 3. Configuration resolution

```bash
bash scripts/train.sh --mode field_component_kl002 --print-plan
bash scripts/train.sh --mode field_component_kl002 --validate-config
```

Save the resolved plan with the run. The validation step must pass before GPU
allocation begins.

For every smoke or formal launch, create a unique `RUN_ID` and a new directory
on a sufficiently large filesystem. Set `OUTPUT_ROOT` and `VF_STORAGE_ROOT` to
that same new location. Never reuse a previous run directory. Formal runs
recommend `VF_MIN_FREE_GIB=500`; a smoke may use a lower host-specific value
after the backing filesystem is checked.

### 4. End-to-end GPU smoke

```bash
export RUN_ID="smoke-field-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}"
export OUTPUT_ROOT="<large-storage-root>/mr-iqa-2/${RUN_ID}"
export VF_STORAGE_ROOT="${OUTPUT_ROOT}"
export VF_MIN_FREE_GIB="<host-appropriate-smoke-threshold>"
bash scripts/train.sh --mode field_component_kl002 --smoke
```

Require one complete 4×36-row trajectory set, one finite optimizer update, and
a valid checkpoint. This verifies the private artifacts and GPU services that
CPU tests cannot inspect.

### 5. Formal epoch chain

Run the full launcher without `--skip-validation` and with another fresh ID and
directory:

```bash
export RUN_ID="field-formal-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}"
export OUTPUT_ROOT="<large-storage-root>/mr-iqa-2/${RUN_ID}"
export VF_STORAGE_ROOT="${OUTPUT_ROOT}"
export VF_MIN_FREE_GIB=500
bash scripts/train.sh --mode field_component_kl002
```

Verify checkpoint steps 291, 582, 873, 1,164, and 1,455. At each boundary the
manifest must show `quarantined → technically_valid → promoted`, and promotion
must follow the complete 200-row, eight-shard Actor→Editor barrier→Judge
`comparison_observational` gate. The next epoch must start from the preceding
checkpoint only through `resolve` on its promoted manifest, rather than from a
directory name or the native Actor.

Retain all three epoch records:

- `state/checkpoints/epochN.json`: `vf_checkpoint_manifest_v2` promotion state
  and evidence;
- `state/checkpoints/epochN.validation.json`: complete observational gate
  summary;
- `state/epochN.json`: `mr_iqa_2_epoch_chain_v2` chain link and final status.

The debugging form `--skip-validation --epochs 1` is the only allowed
skip-validation invocation. It must finish as `technically_valid` and
unpromoted, and it must not be accepted as a parent for another epoch.

### 6. Generalization

```bash
bash scripts/evaluate.sh test
```

Require all six expected row counts, eight complete Actor shards per dataset,
an Editor barrier, no unreported service failures, and a per-row provenance
record. Compare against [`checkpoints.md`](checkpoints.md) only when every
checkpoint and evaluation contract digest matches.

## Per-step evidence

The scientific training record is the merged four-rank trajectory set, not a
single rank's dashboard scalar. For every global step retain:

- the four rank shard IDs and exactly 36 rows per rank;
- reasoning raw reward, rating reward, soft-overlong reward, format status;
- success, actor-ineligible, and service-error counts;
- completion length distribution;
- global or component KL mean/loss and application counts;
- trainer loss and gradient norm;
- optimizer/scheduler update indicators;
- normalized solution diversity and semantic-template rates.

The two published formal runs each contain 1,455 complete global steps and
209,520 trajectories. Across both runs, 419,040 trajectories were audited with
zero missing/duplicate global steps, zero incomplete four-rank steps, and zero
non-finite metric failures. Optimizer and scheduler updated in all 1,455 steps
of each run.

## Checkpoint provenance

Runtime validation and epoch promotion identify a checkpoint with exactly the
ten-file `selected_inference_export` identity. Its digest is:

```text
sha256(sorted(relative_path + NUL + decimal_size + NUL + file_sha256 + LF))
over the 10 allowlisted inference-export files
```

The evaluation contract and `mr_iqa_2_epoch_chain_v2` record the identity as
`checkpoint_digest.{semantics,algorithm,sha256,file_count}`, with
`semantics=selected_inference_export` and `file_count=10`. Optimizer, trainer,
RNG, cache, log, and temporary files never participate in this stable identity.

Publication additionally records a source full-tree hash covering the promoted
training checkpoint and its extra state. The source full tree and selected
public export tree therefore have different scope and normally different
digests. A release manifest should record:

- artifact ID and role;
- parent artifact/tree digest;
- training mode and global step;
- source full checkpoint tree SHA-256;
- each selected public file's relative path, size, and SHA-256;
- public export-tree SHA-256 over the selected allowlist;
- base Actor and source Judge provenance;
- data, prompt, cache, and code contract digests;
- validation status and metrics.

Never infer identity from a directory name such as `best` or `final`. Resolve
the manifest, verify the public files and export tree, and record the immutable
Hugging Face revision. For an epoch chain, only a manifest with
`status=promoted` and `usable=true` may be resolved as the next parent; the
resolver revalidates artifacts and re-hashes the selected export.

For the public source E5 Judge, runtime identity is a two-part check:

```dotenv
JUDGE_MANIFEST_PATH=<repository-root>/checkpoints/mr-iqa-2/judge/source-e5/provenance.json
JUDGE_MODEL_TREE_SHA256=e25415173aacf515e97d5d561c6647a7a84f586061f3a9b2ab3fc079fe21be0a
JUDGE_MODEL_EXPORT_TREE_SHA256=21b232a1a30dc765f3e7cf16c00fd270e4be354615fea0120e32f975e2777e5c
```

The first digest is the source semantic/cache identity recorded by the
manifest. The second is the relocatable integrity digest recomputed over the
downloaded ten-file Hub export. Both must match; they are not interchangeable.

The training/promotion pipeline records the source full-tree digest as a
signed-off lineage statement. The reduced exporter records that provenance but
does not claim to recompute it; it independently constructs and validates the
ten-file public export. A downstream user can verify every public file and the
export-tree digest, but cannot recompute the source full-tree digest from the
reduced snapshot.

Public export-tree digests are:

| Artifact | Public 10-file export-tree SHA-256 |
| --- | --- |
| Field Actor E5 | `3e372f548631e3ebbb23e9d8493cb2d50aa482b1941025deda907b35e0a97edb` |
| Source E5 Judge | `21b232a1a30dc765f3e7cf16c00fd270e4be354615fea0120e32f975e2777e5c` |
| Completion Actor E4 | `fcc36656fd15ba7e164bdf0b0be46290ad231636e88664e7bafaa0982ab59c53` |
| Completion Actor E5 | `14d801bffb7f65217a899b10c0735d3d2e37436dd799c3b6352f085845e5b374` |

## Release verification

The public tutorial was exercised on 2026-08-10 before publication. The CPU
contract suite passed all 106 tests. A fresh eight-GPU
`field_component_kl002 --smoke` run completed one optimizer update through the
real Actor→Editor→Judge path:

| Check | Observed |
| --- | ---: |
| Trajectory shards / rows | 4 × 36 = 144 |
| Optimizer / scheduler updates | 1 / 1 |
| Loss / gradient norm | 0.004201 / 34.86 |
| Mean completion length | 96.07 tokens |
| Reasoning / rating / soft-overlong reward mean | -0.003583 / 0.6783 / 0.0 |
| Global completion KL applications | 0, as required by the field mode |
| ViT / aligner trainable parameters | 0 / 0 |

All reported values were finite; shard identity, trajectory uniqueness, reward
credit, and optimizer/scheduler audits passed. The reasoning/rating component
KL losses were exactly zero on this first step because the policy and fixed
reference Actor were initially identical. No OOM, CUDA, NCCL, or Python
traceback occurred, and all frozen services stopped cleanly after the run.

## Determinism boundary

Seeds and deterministic Judge generation reduce variation, but exact bitwise
reproduction can still depend on GPU model, driver, CUDA libraries, collective
ordering, upstream kernels, and wheel builds. Report the hardware/runtime
inventory and compare semantic invariants, row counts, tree hashes, and metrics
rather than claiming bitwise equivalence without evidence.

If a reproduced value differs, first check model/data/prompt/cache digests,
field eligibility, four-rank completeness, and denominators. Do not explain a
difference as random noise before those contracts match.

## Scientific limitations

The released formal comparison is not a complete factorial experiment. It
compares field credit + component KL with completion-wide credit + global KL,
one seed each. Both credit scope and KL scope change. The 30-step no-KL arms are
pipeline ablations with a shorter horizon and do not fill the two missing
formal factorial cells.

Reproducing the observations supports the reported implementation and
description; it does not turn this design into a single-factor causal estimate.
