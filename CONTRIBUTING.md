# Contributing

Thank you for helping improve MR-IQA-2. Small, auditable changes are preferred
because reward routing, masks, and evaluation order are part of the scientific
contract.

## Before opening a pull request

1. Open an issue for changes that alter the output schema, reward definition,
   KL scope, service order, dataset rows, or checkpoint provenance.
2. Keep machine paths, API keys, model tokens, image roots, caches, generated
   images, and experiment logs outside the repository. Use `.env`, which is
   ignored, and publish only placeholder values in `.env.example`.
3. Do not add model weights or runtime blobs to GitHub. Checkpoint artifacts
   belong in the Hugging Face repository with a file manifest and tree digest.
4. Add or update a contract test for behavior changes.
5. Run the release checks before submitting:

   ```bash
   bash scripts/test_release.sh --static
   bash scripts/test_release.sh
   ```

## Code and experiment changes

- Keep Actor, Editor, Judge, and global orchestration responsibilities
  separate.
- Preserve the structured Actor schema unless the proposal explicitly changes
  it: `reasoning.evidence`, `reasoning.solution`, and `rating`.
- Report changes to credit masks and KL masks independently. A global KL term
  must never be described as a reward term when `kl_in_reward=false`.
- If a change affects the formal mode, record the full configuration, seed,
  data digest, initial model tree digest, source Judge digest, and per-step
  four-rank coverage.
- Do not compare W&B rank-0 reward summaries with globally merged 144-row
  trajectory statistics.
- Treat solution diversity and source-image grounding as first-class metrics;
  rating PLCC/SRCC alone cannot identify solution collapse.

## Documentation and data

Use relative image paths in manifests. Do not contribute dataset images unless
you own the rights and a maintainer has approved their distribution. A manifest
change must update `data/checksums.sha256` and document its source and license.

By contributing, you agree that your contribution is licensed under the MIT
License and that you have the right to submit it.
