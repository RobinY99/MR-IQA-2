from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT.parent


class NativeNoMask30Mask30NoKlContractTests(unittest.TestCase):
    def test_trainer_has_exact_step_boundary_save_and_stop_callback(self) -> None:
        source = (ROOT / "plugin" / "vf_dual_rollout_trainer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("class _StepBoundaryStopCallback(TrainerCallback):", source)
        self.assertIn('os.environ.get("VF_STOP_AFTER_STEP", "")', source)
        self.assertIn("control.should_save = True", source)
        self.assertIn("control.should_training_stop = True", source)
        self.assertIn("self.add_callback(_StepBoundaryStopCallback())", source)

    def test_stage_runner_saves_full_state_and_resumes_in_step_mode(self) -> None:
        source = (
            ROOT / "scripts" / "run_editor_judge_grpo_stage.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('"${MODE}" == "steps"', source)
        self.assertIn('--max_steps "${TRAIN_MAX_STEPS}"', source)
        self.assertIn('--save_strategy steps', source)
        self.assertIn('--save_only_model false', source)
        self.assertIn('--resume_from_checkpoint "${RESUME_CHECKPOINT}"', source)
        self.assertIn('export VF_STOP_AFTER_STEP="${STOP_AFTER_STEP}"', source)

    def test_driver_locks_native_actor_two_masks_and_zero_kl(self) -> None:
        source = (PACKAGE_ROOT / "scripts" / "train.sh").read_text(encoding="utf-8")
        completion = (
            PACKAGE_ROOT / "configs" / "training" / "completion_nokl_30step.env"
        ).read_text(encoding="utf-8")
        field = (
            PACKAGE_ROOT / "configs" / "training" / "field_nokl_30step.env"
        ).read_text(encoding="utf-8")
        self.assertIn('MODEL_PATH="${ACTOR_MODEL_PATH}"', source)
        self.assertIn("TRAIN_MAX_STEPS=30", completion)
        self.assertIn("TRAIN_MAX_STEPS=30", field)
        self.assertIn("COMPONENT_CREDIT_MASK_MODE=completion", completion)
        self.assertIn("COMPONENT_CREDIT_MASK_MODE=field", field)
        self.assertIn("STEP_START=0", completion)
        self.assertIn("STEP_START=0", field)
        for locked_value in (
            "COMPONENT_KL_MODE=off",
            "EXPECTED_COMPONENT_KL_MODE=off",
            "BETA_KL_REASONING=0.0",
            "BETA_KL_RATING=0.0",
            "EXPECTED_BETA_KL_REASONING=0.0",
            "EXPECTED_BETA_KL_RATING=0.0",
            "REFERENCE_ACTIVATION_BETA=0.0",
        ):
            self.assertIn(locked_value, completion)
            self.assertIn(locked_value, field)
        self.assertIn("JUDGER_MAX_NUM_SEQS=1", source)
        self.assertIn("JUDGER_MAX_BATCH_SIZE=1", source)
        self.assertIn("JUDGER_BATCH_WAIT_MS=0", source)

    def test_auditor_requires_full_checkpoint_for_step_mode(self) -> None:
        source = (
            ROOT / "scripts" / "audit_comparison_stage.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'checkpoint_required = config["mode"] in {"formal", "steps"}',
            source,
        )
        self.assertIn('config["mode"] == "steps"', source)

    def test_public_smoke_is_exactly_one_optimizer_update(self) -> None:
        driver = (PACKAGE_ROOT / "scripts" / "train.sh").read_text(
            encoding="utf-8"
        )
        stage = (ROOT / "scripts" / "run_editor_judge_grpo_stage.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('SMOKE_MAX_STEPS="${train_max}"', driver)
        self.assertIn('run_stage "${smoke_dir}" 0 0 1 1 "" "" smoke', driver)
        self.assertIn('SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-5}"', stage)
        self.assertIn('--max_steps "${SMOKE_MAX_STEPS}"', stage)


if __name__ == "__main__":
    unittest.main()
