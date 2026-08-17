# Environment setup

Use separate Python 3.12.13 environments for Actor/Judge and Editor.

## One-command setup

Create both environments for single-image inference:

```bash
bash scripts/setup_envs.sh --profile inference
```

Create and verify the CPU release-test environment:

```bash
bash scripts/setup_envs.sh --profile test
```

Create the training and test environments together after obtaining the
FlashAttention wheel that matches Python, PyTorch, and CUDA:

```bash
FLASH_ATTN_WHEEL=/absolute/path/to/validated_flash_attn.whl \
  bash scripts/setup_envs.sh --profile all
```

Use `--profile training` to omit the CPU test environment, `--no-verify` to
skip post-install checks, or `--dry-run` to inspect every command without
changing the machine. The `training` and `all` profiles create `.env` from
`.env.example` when needed, but local model and dataset paths still have to be
filled in.

## Actor and Judge

```bash
conda env create -f environment/actor-judge.yml
conda activate mr_iqa_actor_judge
python -m pip install --upgrade pip
python -m pip install -r requirements/actor-judge.txt
```

Full training/evaluation requires a CUDA 13.0-compatible driver and eight
visible NVIDIA GPUs.

The launchers also require a prebuilt FlashAttention wheel. Configure it in
your private `.env` using `.env.example`; the launcher validates the artifact
automatically. Install that same wheel into the Actor/Judge environment before
the first preflight:

```bash
python -m pip install /path/to/validated_flash_attn.whl
python -c 'import flash_attn; print(flash_attn.__version__)'
```

The wheel must match the Python, PyTorch, and CUDA ABI.

## Editor

Conda setup:

```bash
conda env create -f environment/editor.yml
conda activate mr_iqa_editor
python -m pip install --upgrade pip
python -m pip install -r requirements/editor.txt
```

Set `DIFFUSERS_VENV` and `DIFFUSERS_MODEL_PATH` in `.env`.

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

Use `bash scripts/test_release.sh --static` before installing test dependencies.
The one-command test profile uses `environment/test.yml` and runs the same
suite after installation.

## Configuration and provenance

Copy `.env.example` to `.env` and fill in local paths. Never commit `.env` or
tokens. Launchers validate configured artifacts automatically.

Capture a sanitized runtime manifest alongside every run without recording a
hostname, username, or local path:

```bash
python environment/capture_runtime.py --role actor-judge > runtime-actor-judge.json
python environment/capture_runtime.py --role editor > runtime-editor.json
```
