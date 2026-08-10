# Global pipeline

`global/` documents the cross-role contracts that connect the Actor, Editor,
and Judge. The executable entry points live in `scripts/` and the
Trainer-coupled reward/KL implementations live in `actor/plugin/` so that
MS-SWIFT can load them as one external plugin.

| Concern | Public implementation |
|---|---|
| Training orchestration | `scripts/train.sh` |
| Validation/generalization | `scripts/evaluate.sh` |
| Four-rank trajectory merge | `actor/plugin/trajectory_io.py` |
| Credit routing | `actor/plugin/credit_assignment.py`, `actor/plugin/token_credit.py` |
| Rating/reasoning rewards | `actor/plugin/margin_reward.py`, `actor/plugin/editor_judge_contract.py` |
| Component/global KL | `actor/plugin/component_loss.py`, `actor/plugin/vf_dual_rollout_trainer.py` |
| Runtime audit | `actor/scripts/audit_comparison_stage.py` |
| Checkpoint provenance | `actor/scripts/checkpoint_manifest.py` |

The directory is intentionally not a Python package: `global` is a Python
keyword. Keeping the actual modules under `actor.plugin` also preserves the
external-plugin import contract used in the released experiments.
