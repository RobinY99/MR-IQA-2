# Orchestration

Use the stable top-level entry points:

```bash
scripts/train.sh --mode completion_global_kl002 --print-plan
scripts/train.sh --mode completion_global_kl002 --validate-config
scripts/train.sh --mode completion_global_kl002 --smoke
scripts/train.sh --mode completion_global_kl002
scripts/evaluate.sh validation
scripts/evaluate.sh test
```

Formal profiles run 291 optimizer updates per epoch and stop at steps 291,
582, 873, 1164, and 1455. With validation enabled, each checkpoint completes
the Actor → Editor barrier → frozen E5 Judge protocol before the next epoch.
