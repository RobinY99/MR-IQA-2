# Third-party notices

The MIT License in this repository applies to original MR-IQA-2 source code and
documentation. It does **not** relicense third-party models, datasets, packages,
or runtime artifacts.

## Models and data

- The Actor starts from the official
  [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)
  model at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, distributed under
  Apache-2.0. The source E5 Judge and released Actor checkpoints are
  derivatives of that base model; their Hugging Face weight release is marked
  Apache-2.0. Downstream users should retain the upstream notices and verify
  the pinned revision.
- The frozen Editor is
  [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294)
  at revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`. It is not distributed or
  trained by this repository. Obtain it independently from that official
  source before setting `DIFFUSERS_MODEL_PATH`. The pinned 4B checkpoint is
  Apache-2.0; retain its upstream license, model-card safety guidance, and
  notices. This repository's MIT License applies only to original MR-IQA-2
  code and does not replace those upstream notices.
- The committed JSONL files are metadata manifests only. Source images are not
  distributed. KonIQ-10K, SPAQ, LIVE-W, KADID-10K, AGIQA-3K, CSIQ, and any
  training-image sources retain their own access and redistribution terms.

## Software dependencies

MR-IQA-2 integrates with PyTorch, Transformers, MS-SWIFT, vLLM, DeepSpeed,
Accelerate, Diffusers, FlashAttention-compatible runtimes, FastAPI, Uvicorn,
NumPy, SciPy, Pillow, Weights & Biases, and other packages listed in
`requirements/`. Each package is governed by its own upstream license and
notices. The pinned version is a reproducibility statement, not a grant of
rights or a guarantee that the package remains secure.

The requirements files download packages from their upstream package indexes;
those packages are not vendored into this GitHub repository. A separately
supplied FlashAttention wheel must be built and distributed in accordance with
its upstream terms.

If a third-party notice appears to be missing or inaccurate, please open an
issue before redistributing a combined package.
