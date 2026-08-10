# Released experiment evidence

This directory contains exported metrics from both five-epoch formal runs.

- `step_metrics_fieldmask_kl002_e5.csv`: all 1,455 optimizer-step metrics for
  field credit + component KL 0.02.
- `step_metrics_completioncredit_globalkl002_e5.csv`: all 1,455 optimizer-step
  metrics for completion-wide credit + global completion KL 0.02.
- `training_epoch_summary.csv`: exact min/mean/max epoch summaries.
- `validation_checkpoint_metrics.csv`: all 10 formal validation checkpoints;
  the machine-specific checkpoint path column is removed.
- `generalization_exact_summary.csv`: exact six-dataset Actor/Editor/Judge
  results for the released Field E5 and Completion E5 checkpoints.
- `collapse_milestones.csv`: measured solution-collapse threshold crossings.

Regenerate them with `scripts/export_public_results.py`; `manifest.json` records
file row counts.
