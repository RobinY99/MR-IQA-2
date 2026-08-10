from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "plugin" / "vf_dual_rollout_trainer.py"
STAGE = ROOT / "scripts" / "run_editor_judge_grpo_stage.sh"
PACKAGE_ROOT = ROOT.parent
FIELD_PROFILE = PACKAGE_ROOT / "configs" / "training" / "field_component_kl002.env"
COMPLETION_PROFILE = (
    PACKAGE_ROOT / "configs" / "training" / "completion_global_kl002.env"
)
AUDITOR = ROOT / "scripts" / "audit_comparison_stage.py"


class MaskedSegmentKlContractTests(unittest.TestCase):
    def test_trainer_uses_independent_masked_sampled_k3_kl_losses(self) -> None:
        source = TRAINER.read_text(encoding="utf-8")
        for token in (
            'component_kl_enabled = bool(self._vf_component_kl_betas)',
            'self._vf_component_kl_betas["reasoning"]',
            'self._vf_component_kl_betas["rating0"]',
            "ref_per_token_logps - per_token_logps",
            "torch.exp(ref_minus_current) - ref_minus_current - 1",
            'component_masks=inputs["vf_component_masks"]',
            "component_betas=self._vf_component_kl_betas",
            "a0_loss = a0_loss + component_kl_total",
            "loss = loss + component_kl_total",
            "elif self.beta != 0.0 and not self.kl_in_reward:",
        ):
            self.assertIn(token, source)
        self.assertNotIn("a0_loss = a0_loss - component_kl_total", source)
        self.assertNotIn("loss = loss - component_kl_total", source)

    def test_stage_locks_exact_betas_reference_and_global_kl_bypass(self) -> None:
        source = STAGE.read_text(encoding="utf-8")
        for token in (
            'COMPONENT_KL_MODE="${COMPONENT_KL_MODE:-off}"',
            'EXPECTED_COMPONENT_KL_MODE="${EXPECTED_COMPONENT_KL_MODE:-${COMPONENT_KL_MODE}}"',
            'EXPECTED_BETA_KL_REASONING="${EXPECTED_BETA_KL_REASONING:-${BETA_KL_REASONING}}"',
            'EXPECTED_BETA_KL_RATING="${EXPECTED_BETA_KL_RATING:-${BETA_KL_RATING}}"',
            '"global_completion_kl_applied": (',
            '"loss_sign": "positive_regularization"',
            'export VF_COMPONENT_KL_MODE="${COMPONENT_KL_MODE}"',
            'export VF_BETA_KL_REASONING="${BETA_KL_REASONING}"',
            'export VF_BETA_KL_RATING="${BETA_KL_RATING}"',
            'export VF_REFERENCE_MODEL_PATH="${REFERENCE_MODEL_PATH}"',
        ):
            self.assertIn(token, source)

    def test_public_profiles_lock_field_and_completion_kl_contracts(self) -> None:
        field = FIELD_PROFILE.read_text(encoding="utf-8")
        completion = COMPLETION_PROFILE.read_text(encoding="utf-8")
        for token in (
            "COMPONENT_CREDIT_MASK_MODE=field",
            "COMPONENT_KL_MODE=field",
            "EXPECTED_COMPONENT_KL_MODE=field",
            "BETA_KL_REASONING=0.02",
            "BETA_KL_RATING=0.02",
            "REFERENCE_ACTIVATION_BETA=0.02",
            "EXPECTED_GLOBAL_COMPLETION_KL_APPLY_COUNT=0",
            "EXPECTED_COMPONENT_KL_APPLY_COUNT=1",
        ):
            self.assertIn(token, field)
        for token in (
            "COMPONENT_CREDIT_MASK_MODE=completion",
            "COMPONENT_KL_MODE=off",
            "BETA_KL_REASONING=0.0",
            "BETA_KL_RATING=0.0",
            "REFERENCE_ACTIVATION_BETA=0.02",
            "EXPECTED_GLOBAL_COMPLETION_KL_APPLY_COUNT=1",
            "EXPECTED_COMPONENT_KL_APPLY_COUNT=0",
            "KL_IN_REWARD=false",
        ):
            self.assertIn(token, completion)

    def test_auditor_requires_per_step_nonnegative_component_kl_metrics(self) -> None:
        source = AUDITOR.read_text(encoding="utf-8")
        for token in (
            'metric("vf/a0_reasoning_kl_loss")',
            'metric("vf/a0_rating0_kl_loss")',
            'metric("vf/global_completion_kl_mean")',
            'metric("vf/global_completion_kl_apply_count")',
            'metric("vf/component_kl_apply_count")',
            '"component_kl_contract": component_kl_contract_ok',
            '"component_kl_metrics": component_kl_metrics_ok',
            '"global_completion_kl_contract": global_completion_kl_contract_ok',
            '"global_completion_kl_metrics": global_completion_kl_metrics_ok',
            "len(reasoning_kl_metrics) == expected_optimizer_updates",
            "len(rating_kl_metrics) == expected_optimizer_updates",
            "math.isfinite(value) and value >= 0.0",
        ):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
