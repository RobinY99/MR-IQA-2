# Environment setup

The validated release uses two isolated Python 3.12.13 environments. The
Actor and frozen Judge share one runtime; the FLUX.2 Editor runs in a second
runtime. This separation is part of the experiment contract.

## Actor and Judge

```bash
conda env create -f environment/actor-judge.yml
conda activate mr_iqa_actor_judge
python -m pip install --upgrade pip
python -m pip install -r requirements/actor-judge.txt
```

The requirements select PyTorch's CUDA 13.0 wheels. A compatible NVIDIA driver
and eight visible GPUs are required for the published full training/evaluation
topology. Smaller hardware configurations are useful for development, but are
not equivalent to the reported run.

The launchers also require a prebuilt FlashAttention wheel. Set
`FLASH_ATTN_WHEEL` and `FLASH_ATTN_WHEEL_SHA256` in your private `.env`; the
launcher validates the artifact automatically. Install that same wheel into
the Actor/Judge environment before the first preflight:

```bash
python -m pip install /path/to/validated_flash_attn.whl
python -c 'import flash_attn; print(flash_attn.__version__)'
```

The wheel must match the published Python, PyTorch, and CUDA ABI. The repository
does not redistribute it and does not compile an unpinned replacement during a
run.

## Editor

Conda setup:

```bash
conda env create -f environment/editor.yml
conda activate mr_iqa_editor
python -m pip install --upgrade pip
python -m pip install -r requirements/editor.txt
```

Alternatively, create a Python 3.12 virtual environment and install the same
requirements file. Set `DIFFUSERS_VENV` to that environment and
`DIFFUSERS_MODEL_PATH` to the downloaded Editor checkpoint.

## CPU release checks

The repository's format, privacy, data-integrity, and contract tests do not
load model weights or initialize CUDA:

```bash
python -m venv .venv/release-test
source .venv/release-test/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/test.txt
python -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cpu
bash scripts/test_release.sh
```

Use `bash scripts/test_release.sh --static` when PyTorch or Pillow are not yet
installed. The static mode still checks JSONL integrity and schemas, Python
and shell syntax, forbidden checkpoint blobs, unsafe symlinks, and common
private path or credential patterns.

## Configuration and provenance

Copy `.env.example` to `.env` and fill in local paths. Never commit `.env`, API
tokens, private model paths, generated images, or original-score caches. Each
published checkpoint is validated against its Hugging Face manifest by the
launcher/preflight before training or evaluation.

Capture a sanitized runtime manifest alongside every run without recording a
hostname, username, or local path:

```bash
python environment/capture_runtime.py --role actor-judge > runtime-actor-judge.json
python environment/capture_runtime.py --role editor > runtime-editor.json
```
