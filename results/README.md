# Released experiment evidence

This directory contains path-free, derived evidence from the two five-epoch
formal runs. It is provided so reported behavior can be audited without access
to private infrastructure or raw images.

- `step_metrics_fieldmask_kl002_e5.csv`: all 1,455 optimizer-step metrics for
  field credit + component KL 0.02.
- `step_metrics_completioncredit_globalkl002_e5.csv`: all 1,455 optimizer-step
  metrics for completion-wide credit + global completion KL 0.02.
- `training_epoch_summary.csv`: exact min/mean/max epoch summaries.
- `validation_checkpoint_metrics.csv`: all 10 formal validation checkpoints;
  the machine-specific checkpoint path column is removed.
- `generalization_exact_summary.csv`: exact six-dataset Actor/Editor/Judge
  results for the released Field E5 and Completion E4/E5 checkpoints.
- `collapse_milestones.csv`: measured solution-collapse threshold crossings.

The files are regenerated with `scripts/export_public_results.py`. The export
drops absolute checkpoint/model/data paths but does not alter metric values.
`manifest.json` records row counts and SHA256 hashes.

The checkpoint tree column in `validation_checkpoint_metrics.csv` is the
**source full checkpoint promotion digest** from the training run. It is not
the tree digest of the reduced ten-file Hugging Face export. Public export-tree
digests and their source-to-export mapping are documented in
[`../docs/checkpoints.md`](../docs/checkpoints.md) and the Hugging Face
`checkpoint_manifest.json`.
