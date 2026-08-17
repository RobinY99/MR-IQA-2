<p align="center">
  <img src="assets/logo.png" alt="MR-IQA-2 logo" width="260">
</p>

# MR-IQA-2: Masked Credit Assignment for Visual Quality Reasoning

<p align="center">
  <a href="paper/README.md">Paper source</a> |
  <a href="https://huggingface.co/RobinY99/MR-IQA-2">Model weights</a> |
  <a href="docs/training.md">Training</a> |
  <a href="docs/evaluation.md">Evaluation</a>
</p>

MR-IQA-2 trains a multimodal Actor to produce image-quality evidence, an edit
solution, and a numeric rating. A frozen FLUX.2-klein-4B Editor applies the
solution, and a frozen E5 Judge measures the quality change. The released Actor
uses masked credit so that component rewards are assigned through masks over
their eligible generated tokens.

The Hugging Face release contains exactly three models:

- `actor/`: masked credit E5 Actor, step 1,455;
- `judge/`: frozen E5 Judge, step 725;
- `editor/`: FLUX.2-klein-4B.

## Repository contents

- `actor/`: Actor training, rollout, checkpoint, and evaluation code.
- `editor/`: FLUX.2-klein service and client.
- `judge/`: frozen E5 Judge service and scoring tools.
- `global/`: shared reward, KL, and runtime contracts.
- `configs/training/`: formal training presets.
- `data/`: 7,000 training rows, 200 validation rows, and 28,270 test rows.
- `paper/`: merged arXiv manuscript and standalone supplementary materials.
- `environment/`, `requirements/`, `scripts/`: environments, launchers, tests,
  and release tools.

## Installation

Full training and evaluation require Linux, CUDA 13.0, and one host with eight
visible NVIDIA GPUs.

Clone the repository, then create the two pinned inference environments with
one command:

```bash
git clone https://github.com/RobinY99/MR-IQA-2.git
cd MR-IQA-2
bash scripts/setup_envs.sh --profile inference
```

The same entry point can create and verify the CPU test environment:

```bash
bash scripts/setup_envs.sh --profile test
```

For full training plus tests, provide the ABI-compatible FlashAttention wheel
documented in [`environment/README.md`](environment/README.md):

```bash
FLASH_ATTN_WHEEL=/absolute/path/to/validated_flash_attn.whl \
  bash scripts/setup_envs.sh --profile all
```

The `training` and `all` profiles also create a private `.env` from
`.env.example` when one does not exist. Fill in the machine-local model, data,
and output paths before launching a run. The setup command installs software;
it does not download model weights or datasets.

Download the base model and the three released models:

```bash
python -m pip install -r requirements/publish.txt

huggingface-cli download Qwen/Qwen3.5-4B \
  --local-dir checkpoints/qwen3.5-4b

huggingface-cli download RobinY99/MR-IQA-2 \
  --include "actor/**" "judge/**" "editor/**" \
  --local-dir checkpoints/mr-iqa-2
```

Set the local paths in `.env`:

```dotenv
ACTOR_MODEL_PATH=<repository-root>/checkpoints/qwen3.5-4b
DIFFUSERS_MODEL_PATH=<repository-root>/checkpoints/mr-iqa-2/editor
JUDGE_MODEL_PATH=<repository-root>/checkpoints/mr-iqa-2/judge
JUDGE_MANIFEST_PATH=<repository-root>/checkpoints/mr-iqa-2/judge/provenance.json
ORIGINAL_SCORE_CACHE_PATH=<local-generated-j0-cache.sqlite>
```

Keep the remaining variables from `.env.example`. Generate the deterministic
J0 cache as described in [`docs/training.md`](docs/training.md).

## Loading released models

Actor and Judge load directly from their Hugging Face subfolders:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

repo_id = "RobinY99/MR-IQA-2"


def load_qwen(role):
    processor = AutoProcessor.from_pretrained(
        repo_id,
        subfolder=role,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        repo_id,
        subfolder=role,
        torch_dtype="auto",
        device_map="auto",
        use_safetensors=True,
    )
    return processor, model


actor_processor, actor = load_qwen("actor")
judge_processor, judge = load_qwen("judge")
```

The Editor loads from its downloaded subfolder:

```python
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from huggingface_hub import snapshot_download

snapshot = snapshot_download(
    "RobinY99/MR-IQA-2",
    allow_patterns=["editor/**"],
)
editor = Flux2KleinPipeline.from_pretrained(
    Path(snapshot) / "editor",
    torch_dtype=torch.bfloat16,
)
```

## Quick start: one image, one GPU

After installing the two validated environments above, provide one input image:

```bash
python examples/quick_start.py /absolute/path/to/input.jpg
```

The script uses GPU 0 by default. It runs the Actor, Editor, and frozen E5
Judge sequentially so only one model occupies the selected GPU at a time. The
Judge scores both the input and edited images and reports `J0`, `J1`, and
`J1-J0`. No HTTP service is started. Use another GPU or cached models with:

```bash
python examples/quick_start.py /absolute/path/to/input.jpg \
  --gpu 1 \
  --local-files-only \
  --output-dir outputs/my_image
```

The output directory contains `actor_raw.txt`, `assessment.json`, `edited.png`,
`evaluation.json`, and `result.json`. Actor/Judge and Editor remain in separate
processes because their validated dependency versions differ.

## Single-image Actor to Editor example

[`examples/actor_to_editor.py`](examples/actor_to_editor.py) runs the released
Actor with the same neutral prompt contract used for validation, validates the
structured completion, and forwards its `reasoning.solution` unchanged to a
local FLUX.2 Klein Editor service. This example intentionally stops after the
edit and does not run the Judge.

Start one Editor service on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n mr_iqa_editor \
  python -m editor.server \
  --host 127.0.0.1 \
  --port 8212 \
  --model-path checkpoints/mr-iqa-2/editor \
  --output-dir outputs/actor_to_editor/editor
```

After `/health` reports `ready: true`, run the Actor on GPU 0. Both processes
must see the same absolute image path.

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n mr_iqa_actor_judge \
  python examples/actor_to_editor.py \
  --image /absolute/path/to/source.png \
  --actor-model checkpoints/mr-iqa-2/actor \
  --local-files-only \
  --editor-url http://127.0.0.1:8212 \
  --device cuda:0 \
  --output-dir outputs/actor_to_editor/sample_0001
```

The output directory contains the exact Actor completion and parsed JSON, the
Editor request and response, source and edited images, and a provenance file.
A real output from this command is published in the Hugging Face
[`sample_0001` assets](https://huggingface.co/RobinY99/MR-IQA-2/tree/main/assets/actor_editor/sample_0001)
and its
[`sample_0001.json`](https://huggingface.co/RobinY99/MR-IQA-2/blob/main/examples/actor_editor/sample_0001.json).

## Eight-GPU deployment

The launchers use a fixed single-host topology. During training, four Actor
ranks run on GPUs 0–3 while each GPU in 4–7 hosts one Editor service and one
Judge service:

| GPU | Training process | Editor URL | Judge URL |
| ---: | --- | --- | --- |
| 0–3 | Actor ranks 0–3 | — | — |
| 4 | Editor + Judge lane 0 | `http://127.0.0.1:8212` | `http://127.0.0.1:8204` |
| 5 | Editor + Judge lane 1 | `http://127.0.0.1:8213` | `http://127.0.0.1:8205` |
| 6 | Editor + Judge lane 2 | `http://127.0.0.1:8214` | `http://127.0.0.1:8206` |
| 7 | Editor + Judge lane 3 | `http://127.0.0.1:8215` | `http://127.0.0.1:8207` |

`scripts/train.sh` starts, checks, routes requests to, and stops these owned
services. Each optimizer update merges 36 trajectories from each Actor rank,
for 144 trajectories globally. The ViT and multimodal aligner remain frozen.

Full evaluation reuses all eight GPUs in three non-overlapping phases:

1. Actor inference runs eight shards on GPUs 0–7 and merges their outputs.
2. Editor inference runs eight lanes on GPUs 0–7 at ports 8212–8219. Every
   edit must finish before the Editor services are stopped.
3. Judge inference runs eight lanes on GPUs 0–7 at ports 8204–8211 and scores
   the saved edited images.

By default, the Actor evaluation workers use temporary internal vLLM ports
40000, 41024, 42048, 43072, 44096, 45120, 46144, and 47168. They are not
Editor/Judge HTTP endpoints. `scripts/evaluate.sh` waits for all eight GPUs
before Actor inference and before entering the offline service stages. The
offline runner enforces the Editor barrier, stops Editor, and only then starts
Judge.

Print the resolved deployment without loading a model:

```bash
bash scripts/train.sh --mode field_component_kl002 --print-plan
bash scripts/evaluate.sh --print-plan
```

## Port and HTTP interaction

Editor and Judge services bind to `127.0.0.1`; they are available only to
processes on the same host. Requests pass absolute local image paths, not image
bytes, so the caller and service must see the same filesystem.

| Service | Method and endpoint | Request | Main response fields |
| --- | --- | --- | --- |
| Editor | `GET /health` | none | `ready`, `backend`, model/runtime metadata |
| Editor | `POST /edit` | `image_path` and at least one of `positive_prompt` or `region_prompt`; optional `negative_prompt`, `edit_plan`, `request_index`, `seed` | `status`, `edited_path`, `seed`, runtime metadata |
| Judge | `GET /health` | none | `ready`, model and generation provenance |
| Judge | `POST /score_image` | `image_path`; optional `repeats` in 1–4 | `status`, `mean`, `valid_count`, `outputs` |

`negative_prompt` is accepted for request compatibility, but the current
Editor profile reports `negative_prompt_supported: false`.

After a launcher reports that the services are ready, inspect them with:

```bash
curl -fsS http://127.0.0.1:8212/health | python -m json.tool
curl -fsS http://127.0.0.1:8204/health | python -m json.tool
```

An Editor request returns an `edited_path` on the local filesystem:

```bash
curl -fsS -X POST http://127.0.0.1:8212/edit \
  -H 'Content-Type: application/json' \
  -d '{
    "image_path": "/absolute/path/to/input.png",
    "positive_prompt": "Reduce noise while preserving content and geometry.",
    "request_index": 0
  }' | python -m json.tool
```

Pass that path to the Judge:

```bash
curl -fsS -X POST http://127.0.0.1:8204/score_image \
  -H 'Content-Type: application/json' \
  -d '{
    "image_path": "/absolute/path/to/edited.png",
    "repeats": 1
  }' | python -m json.tool
```

Do not expose these ports publicly. The published launchers treat them as a
fixed contract. If adapting the topology in code, update the GPU lists, service
ports, and Actor-facing URL lists together; the service manager rejects
duplicate ports, mismatched lane counts, and ports outside 1024–65535.

## Masked credit training

The released formal setting uses the launcher preset
`field_component_kl002`: masked credit, reasoning and rating component KL with
beta 0.02, five epochs, and 291 optimizer updates per epoch. Validation covers
all 200 rows between epochs, and only a promoted checkpoint can seed the next
epoch.

Validate the configuration before loading a model:

```bash
bash scripts/test_release.sh --static
bash scripts/train.sh --mode field_component_kl002 --validate-config
```

The launchers source `ENV_FILE` and values in that file take precedence over
same-named shell variables. Create a smoke-specific copy when changing its free
space threshold, then use a new ID and output directory for every invocation:

```bash
cp .env .env.smoke
# Edit .env.smoke and set VF_MIN_FREE_GIB for this host.

export RUN_ID="smoke-masked-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}"
export OUTPUT_ROOT="<large-storage-root>/mr-iqa-2/${RUN_ID}"
export VF_STORAGE_ROOT="${OUTPUT_ROOT}"
ENV_FILE="$PWD/.env.smoke" \
  bash scripts/train.sh --mode field_component_kl002 --smoke

export RUN_ID="masked-formal-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}"
export OUTPUT_ROOT="<large-storage-root>/mr-iqa-2/${RUN_ID}"
export VF_STORAGE_ROOT="${OUTPUT_ROOT}"
bash scripts/train.sh --mode field_component_kl002
```

`WANDB_MODE=offline` is the default. See
[`docs/training.md`](docs/training.md) for data, cache, checkpoint, and resume
contracts.

## Evaluation

Create a dedicated evaluation environment so the evaluated checkpoint and
image root are explicit:

```bash
cp .env .env.eval
# In .env.eval, set:
# ACTOR_MODEL_PATH=<actor-checkpoint>
# EVAL_IMAGE_ROOT=<dataset-image-root>
```

Actor-only evaluation measures output validity and rating PLCC/SRCC/MAE
without starting the Editor or Judge:

```bash
ENV_FILE="$PWD/.env.eval" EVAL_ACTOR_ONLY=1 \
  bash scripts/evaluate.sh test
```

Full validation and six-dataset evaluation use the eight-GPU phases described
above:

```bash
ENV_FILE="$PWD/.env.eval" bash scripts/evaluate.sh validation
ENV_FILE="$PWD/.env.eval" bash scripts/evaluate.sh test
ENV_FILE="$PWD/.env.eval" bash scripts/evaluate.sh all
```

The `all` mode runs validation plus all six test datasets. See
[`docs/evaluation.md`](docs/evaluation.md) for output schemas and resume rules.

## PLCC/SRCC performance

Actor-only rating performance on the six generalization datasets is shown
below. Each entry is `PLCC / SRCC`; Average is the unweighted macro mean of
the six dataset-level coefficients.

| Model | KonIQ-10K | SPAQ | LIVE-W | AGIQA-3K | KADID-10K | CSIQ | Average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [MR-IQA](https://github.com/RobinY99/MR-IQA) | 0.949 / 0.931 | 0.892 / 0.897 | 0.899 / 0.883 | 0.804 / 0.732 | 0.672 / 0.683 | 0.767 / 0.732 | 0.831 / 0.810 |
| MR-IQA-2 | 0.937 / 0.917 | 0.900 / 0.899 | 0.893 / 0.863 | 0.809 / 0.739 | 0.667 / 0.669 | 0.824 / 0.785 | 0.838 / 0.812 |

MR-IQA values are from the released Qwen3-VL-2B result in the
[MR-IQA paper](https://arxiv.org/pdf/2606.29760). MR-IQA-2 values use the
released masked-credit E5 Actor at step 1,455; exact valid-row counts and
unrounded coefficients are reported in
[`docs/checkpoints.md`](docs/checkpoints.md#field-e5-recommended).

## Verification

```bash
# Dependency-free schemas, syntax, privacy, and artifact checks.
bash scripts/test_release.sh --static

# Full CPU contract suite after installing requirements/test.txt and CPU PyTorch.
bash scripts/test_release.sh
```

Training checks shard completeness, update counts, finite rewards and KL, and
the expected masked credit and component-KL application counts.

## License and citation

Original code is MIT licensed. Model weights, datasets, dependencies, and the
frozen Editor retain their upstream terms. Qwen-derived Actor/Judge weights
and the pinned FLUX.2-klein-4B Editor are Apache-2.0. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Please cite the software metadata in [`CITATION.cff`](CITATION.cff). If you use
MR-IQA as well, cite its associated paper and repository separately.
