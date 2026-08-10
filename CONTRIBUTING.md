# Contributing

## Before opening a pull request

1. Open an issue before changing schemas, rewards, KL scope, service order,
   datasets, or checkpoint provenance.
2. Keep secrets, private paths, images, caches, weights, and logs out of Git.
3. Add or update contract tests.
4. Run:

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
- Record the configuration, seed, artifact versions, Judge contract, and
  four-rank coverage for formal-mode changes.
- Do not compare W&B rank-0 reward summaries with globally merged 144-row
  trajectory statistics.

## Documentation and data

Use relative image paths. Add images only with distribution rights and
maintainer approval; document each manifest source and license.

By contributing, you agree that your contribution is licensed under the MIT
License and that you have the right to submit it.
