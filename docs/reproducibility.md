# Reproducibility contract

Reproduction uses pinned environments, relative manifests, explicit launchers,
and validated service, reward, KL, and checkpoint-chain contracts.

## Fixed formal configuration

| Property | Published value |
| --- | --- |
| Initial Actor repository | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a), Apache-2.0 |
| Initial Actor Hub revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Frozen Editor repository | [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294) |
| Frozen Editor Hub revision | `e7b7dc27f91deacad38e78976d1f2b499d76a294` |
| Judge provenance manifest | `judge/source-e5/provenance.json` |
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

Use `scripts/train.sh --print-plan` to resolve mode-specific values.

## Validation ladder

Run checks from least expensive to most expensive:

### 1. Dependency-free source release check

```bash
bash scripts/test_release.sh --static
```

### 2. CPU contract tests

Create the lightweight test environment described in
[`../environment/README.md`](../environment/README.md), then run:

```bash
bash scripts/test_release.sh
```

### 3. Configuration resolution

```bash
bash scripts/train.sh --mode field_component_kl002 --print-plan
bash scripts/train.sh --mode field_component_kl002 --validate-config
```

Save the plan. Use a unique `RUN_ID`; set `OUTPUT_ROOT` and `VF_STORAGE_ROOT` to
the same new directory.

### 4. End-to-end GPU smoke

```bash
export RUN_ID="smoke-field-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}"
export OUTPUT_ROOT="<large-storage-root>/mr-iqa-2/${RUN_ID}"
export VF_STORAGE_ROOT="${OUTPUT_ROOT}"
export VF_MIN_FREE_GIB="<host-appropriate-smoke-threshold>"
bash scripts/train.sh --mode field_component_kl002 --smoke
```

Require 4×36 rows, one finite optimizer update, and a valid checkpoint.

### 5. Formal epoch chain

Run the full launcher with another fresh ID and directory:

```bash
export RUN_ID="field-formal-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}"
export OUTPUT_ROOT="<large-storage-root>/mr-iqa-2/${RUN_ID}"
export VF_STORAGE_ROOT="${OUTPUT_ROOT}"
export VF_MIN_FREE_GIB=500
bash scripts/train.sh --mode field_component_kl002
```

Verify steps 291, 582, 873, 1,164, and 1,455. Each must pass the complete
200-row, eight-shard Actor→Editor barrier→Judge gate before promotion; the next
epoch resolves that promoted manifest.

Retain all three epoch records:

- `state/checkpoints/epochN.json`: `vf_checkpoint_manifest_v2` promotion state
  and evidence;
- `state/checkpoints/epochN.validation.json`: complete observational gate
  summary;
- `state/epochN.json`: `mr_iqa_2_epoch_chain_v2` chain link and final status.

`--skip-validation --epochs 1` leaves an unpromoted checkpoint that cannot seed
another epoch.

### 6. Generalization

```bash
bash scripts/evaluate.sh test
```

Require all six row counts, eight Actor shards per dataset, an Editor barrier,
and no unreported service failures.

## Per-step evidence

For every global step, retain the merged four-rank trajectory fields:

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

Only a manifest with `status=promoted` and `usable=true` may seed the next
epoch. Launchers revalidate it automatically.

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

All reported values were finite. Shard, reward, optimizer, scheduler, and
service audits passed. The first-step reasoning/rating KL losses were exactly
zero because policy and reference Actor were initially identical.
