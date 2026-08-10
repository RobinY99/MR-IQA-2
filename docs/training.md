# Training guide

These commands train only the Actor; Editor and E5 Judge remain frozen.

## 1. Hardware and software

The formal topology requires Linux, a CUDA 13.0-compatible driver, and eight
visible NVIDIA GPUs:

- GPUs 0–3: four Actor ranks;
- GPUs 4–7: four Editor/Judge service lanes;
- 144 trajectories per optimizer update;
- 291 optimizer updates per formal epoch;
- five epochs, ending at global step 1,455.

Create the two Python 3.12.13 environments in
[`../environment/README.md`](../environment/README.md).

## 2. Required artifacts

Obtain the following before training:

1. the Qwen3.5-4B-compatible initial Actor;
2. `judge/`, the frozen E5 Judge and its provenance file;
3. `editor/`, the frozen FLUX.2-klein-4B Editor;
4. source images corresponding to `data/train.jsonl`;
5. a locally built original-image Judge score cache;
6. a compatible prebuilt FlashAttention wheel.

Project weights are at
[huggingface.co/RobinY99/MR-IQA-2](https://huggingface.co/RobinY99/MR-IQA-2).
The released mask E5 Actor is in `actor/`; formal reproduction from epoch 1
starts from the pinned Qwen base model below.

### Download the models

The initial Actor is the official Apache-2.0
[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)
at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`:

```bash
python -m pip install -r requirements/publish.txt
huggingface-cli download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir checkpoints/qwen3.5-4b
```

Download the frozen Judge and Editor from the project repository:

```bash
huggingface-cli download RobinY99/MR-IQA-2 \
  --revision 402afd29be9eb539d9d6b054a985cb8c49c32bd5 \
  --include "judge/**" "editor/**" \
  --local-dir checkpoints/mr-iqa-2
```

The Editor is the exact
[`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294)
revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`. Diffusers must receive
the local `editor/` directory. To obtain only that directory:

```python
from pathlib import Path
import torch
from diffusers import Flux2KleinPipeline
from huggingface_hub import snapshot_download

snapshot = snapshot_download(
    "RobinY99/MR-IQA-2",
    revision="402afd29be9eb539d9d6b054a985cb8c49c32bd5",
    allow_patterns=["editor/**"],
)
editor_path = Path(snapshot) / "editor"
editor = Flux2KleinPipeline.from_pretrained(
    editor_path,
    torch_dtype=torch.bfloat16,
)
```

### Build the original-image J0 cache locally

The model repository does not contain a training cache. After configuring
`.env`, build the 7,000-row source manifest, score it with the frozen Judge,
then create the local cache:

```bash
set -a
source .env
set +a

CACHE_OUTPUT=/path/to/local-output
SOURCE_MANIFEST="${CACHE_OUTPUT}/source-image-manifest.jsonl"
JUDGE_RUN_DIR="${CACHE_OUTPUT}/judge-service"
mkdir -p "${CACHE_OUTPUT}"

"${ACTOR_PYTHON}" scripts/build_source_manifest.py \
  --train-manifest data/train.jsonl \
  --image-root "${TRAIN_IMAGE_ROOT}" \
  --output "${SOURCE_MANIFEST}" \
  --expected-samples 7000

bash judge/launch.sh start "${JUDGE_RUN_DIR}"
"${JUDGER_PYTHON}" judge/score_manifest.py \
  --source-manifest "${SOURCE_MANIFEST}" \
  --output "${CACHE_OUTPUT}/deterministic-source-judge-output.jsonl" \
  --host 127.0.0.1 \
  --ports "${JUDGER_PORTS:-8204,8205,8206,8207}" \
  --resume
bash judge/launch.sh stop "${JUDGE_RUN_DIR}"

"${ACTOR_PYTHON}" actor/scripts/build_e5_original_score_cache.py \
  --source-manifest "${SOURCE_MANIFEST}" \
  --judge-scores "${CACHE_OUTPUT}/deterministic-source-judge-output.jsonl" \
  --output-sqlite "${CACHE_OUTPUT}/original_score_cache.sqlite" \
  --output-summary "${CACHE_OUTPUT}/original_score_cache.summary.json" \
  --judge-model-id "${JUDGE_MODEL_ID:-source-e5-judge-step725}" \
  --judge-model-path "${JUDGE_MODEL_PATH}" \
  --judge-model-tree-sha256 "${JUDGE_MODEL_TREE_SHA256}" \
  --judge-prompt-schema "${JUDGER_PROMPT_SCHEMA:-e5_training_reasoning_v5}" \
  --judge-prompt-hash "${JUDGE_PROMPT_HASH}" \
  --payload-schema vf_original_score_cache_e5_judge_v1 \
  --cache-actor-id source-e5-judge-step725-original-score \
  --expected-samples 7000
```

The scoring command writes one atomic JSONL row per source sample. Failed rows
remain explicit and return a nonzero status; rerunning with `--resume` retains
successful rows and retries the rest.

The source-manifest builder records the absolute image path, SHA-256, width,
and height for every unique training sample and rejects missing or duplicate
images. Set `ORIGINAL_SCORE_CACHE_PATH` and copy `sqlite_sha256` from the
generated summary into `ORIGINAL_SCORE_CACHE_SHA256`. Do not reuse a cache
built with a different Judge, prompt, image manifest, or image contents.

## 3. Private machine configuration

```bash
cp .env.example .env
```

Fill `.env` with:

- Conda initialization and Actor/Judge Python;
- initial Actor path and manifest identity;
- training and evaluation image roots;
- Editor environment and model path;
- Judge model, manifest, prompt schema, and identity fields;
- original-score cache path, row/sample contract, schema, and identity field;
- FlashAttention wheel path and artifact identity.

Configure the downloaded models:

```dotenv
DIFFUSERS_MODEL_PATH=<repository-root>/checkpoints/mr-iqa-2/editor
JUDGE_MODEL_PATH=<repository-root>/checkpoints/mr-iqa-2/judge
JUDGE_MANIFEST_PATH=<repository-root>/checkpoints/mr-iqa-2/judge/provenance.json
```

Configure the J0 cache:

```dotenv
ORIGINAL_SCORE_CACHE_PATH=<local-output>/original_score_cache.sqlite
ORIGINAL_SCORE_CACHE_SHA256=<sqlite_sha256-from-summary>
ORIGINAL_SCORE_CACHE_EXPECTED_ROW_COUNT=7000
ORIGINAL_SCORE_CACHE_EXPECTED_SAMPLE_COUNT=7000
ORIGINAL_SCORE_CACHE_EXPECTED_ACTOR_IDS=source-e5-judge-step725-original-score
ORIGINAL_SCORE_CACHE_PAYLOAD_SCHEMA=vf_original_score_cache_e5_judge_v1
ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MIN=0.0
ORIGINAL_SCORE_CACHE_EXPECTED_RATING_MAX=5.0
```

## 4. Inspect and validate a mode

List the exact contract without loading models:

```bash
bash scripts/train.sh --mode field_component_kl002 --print-plan
bash scripts/train.sh --mode completion_global_kl002 --print-plan
```

Validate the configuration:

```bash
bash scripts/train.sh --mode field_component_kl002 --validate-config
```

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

Use a unique `RUN_ID`; `OUTPUT_ROOT` and `VF_STORAGE_ROOT` must be the same new
directory. `WANDB_MODE=offline` is the default.

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

Promotion requires all 200 source rows, eight Actor shards, valid schema,
complete Editor/Judge records, zero service errors, and an Editor barrier.

### Checkpoint promotion records

Each epoch writes three related records below `OUTPUT_ROOT`:

| Path | Purpose |
| --- | --- |
| `state/checkpoints/epochN.json` | Promotion state and checkpoint identity |
| `state/checkpoints/epochN.validation.json` | Complete 200-row validation evidence |
| `state/epochN.json` | Epoch-chain record |

Use `--epochs N` for a deliberate prefix of one to five epochs. Use
`--skip-validation` only with `--epochs 1`; the launcher rejects every other
combination. This debugging path stops after technical validation, leaves the
checkpoint unpromoted, and cannot seed another epoch.

## 7. Thirty-step ablations

The two independent no-KL probes both start from the native Actor:

```bash
bash scripts/train.sh --mode field_nokl_30step
bash scripts/train.sh --mode completion_nokl_30step
```

Both start from the native Actor.

## 8. Required runtime audits

A valid formal update requires:

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

Aggregate rewards from all four ranks; W&B `vf/*reward_mean` is rank-0 local.

## 9. Monitoring collapse

Retain all 200 validation rows and monitor:

- exact and normalized solution uniqueness;
- modal normalized-solution share;
- semantic template-family share;
- evidence uniqueness;
- success, actor-ineligible, and service-error counts;
- `J0`, `J1`, delta, and zero-filled reasoning reward.

Completion-global crossed 50% house-template share at step 236 and 90% at step
265; E1 produced house-family solutions for all 197 eligible validation rows.

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

Resume only from the validated promoted parent with the same mode, data, Judge,
prompt, and initial/reference Actor.
