# Requirements

MR-IQA-2 uses separate environments because the Actor/Judge and Editor stacks
have different validated `transformers` and `accelerate` versions. Mixing these
files into one environment is unsupported.

| File | Purpose | GPU required |
| --- | --- | --- |
| `actor-judge.txt` | GRPO Actor training, Actor inference, and frozen E5 Judge | NVIDIA CUDA |
| `editor.txt` | FLUX.2 image-editing service | NVIDIA CUDA |
| `test.txt` | Static checks and CPU contract tests | No |
| `publish.txt` | Hugging Face upload tooling | No |

The production files use the CUDA 13.0 PyTorch wheel index. FlashAttention is
deliberately not downloaded by either requirements file: the training launcher
requires a prebuilt, validated wheel and checks it automatically before use.
This prevents an unreviewed local compilation from silently changing the
runtime. Install that wheel explicitly into the Actor/Judge environment, then
configure it in the private `.env` using `.env.example`:

```bash
python -m pip install /path/to/validated_flash_attn.whl
```

For a CPU-only development environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements/test.txt
python -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cpu
bash scripts/test_release.sh
```

The CPU wheel matches the published PyTorch version but omits CUDA libraries.
See `environment/README.md` for the exact GPU environment setup.
