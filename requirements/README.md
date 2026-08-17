# Requirements

Actor/Judge and Editor use separate environments.

Install and verify the pinned environments through the public wrapper:

```bash
bash scripts/setup_envs.sh --profile inference
bash scripts/setup_envs.sh --profile test
```

Full training additionally requires `FLASH_ATTN_WHEEL`; see
`environment/README.md` for the one-command form.

| File | Purpose | GPU required |
| --- | --- | --- |
| `actor-judge.txt` | GRPO Actor training, Actor inference, and frozen E5 Judge | NVIDIA CUDA |
| `editor.txt` | FLUX.2 image-editing service | NVIDIA CUDA |
| `test.txt` | Static checks and CPU contract tests | No |
| `publish.txt` | Hugging Face upload tooling | No |

Install the compatible FlashAttention wheel into Actor/Judge and configure it
through `.env.example`:

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

See `environment/README.md` for the exact GPU environment setup.
