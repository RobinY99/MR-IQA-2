from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # Local macOS system Python; exercised in the remote training env.
    torch = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))

if torch is not None:
    from component_loss import (  # noqa: E402
        combine_active_branch_losses,
        compute_component_kl_loss,
        compute_component_policy_loss,
    )


@unittest.skipIf(torch is None, "PyTorch is only available in the remote training environment")
class ComponentLossTests(unittest.TestCase):
    def test_each_component_is_normalized_by_its_own_token_mask(self) -> None:
        ratio = torch.ones((1, 8))
        completion_mask = torch.ones((1, 8), dtype=torch.bool)
        masks = {
            "rating0": torch.tensor([[False, False, True, False, False, False, False, False]]),
            "edit_gain": torch.tensor([[False, False, False, True, True, True, False, False]]),
        }
        advantages = {
            "rating0": torch.tensor([1.0]),
            "edit_gain": torch.tensor([1.0]),
        }
        total, components, active_count = compute_component_policy_loss(
            ratio=ratio,
            clipped_ratio=ratio,
            rollout_is_weights=ratio,
            completion_mask=completion_mask,
            component_masks=masks,
            component_advantages=advantages,
            component_weights={"rating0": 1.0, "edit_gain": 1.0},
        )
        self.assertEqual(active_count, 1)
        self.assertAlmostEqual(float(components["rating0"]), -1.0)
        self.assertAlmostEqual(float(components["edit_gain"]), -1.0)
        self.assertAlmostEqual(float(total), -2.0)

    def test_branch_combination_is_weighted_by_active_rows(self) -> None:
        combined = combine_active_branch_losses(
            [(torch.tensor(2.0), 4), (torch.tensor(10.0), 1)]
        )
        self.assertAlmostEqual(float(combined), 3.6, places=6)

    def test_inactive_nonfinite_tokens_cannot_poison_loss_or_gradient(self) -> None:
        ratio = torch.tensor([[1.0, float("inf")]], requires_grad=True)
        completion_mask = torch.ones((1, 2), dtype=torch.bool)
        total, _, _ = compute_component_policy_loss(
            ratio=ratio,
            clipped_ratio=torch.tensor([[1.0, 1.2]]),
            rollout_is_weights=torch.tensor([[1.0, float("inf")]]),
            completion_mask=completion_mask,
            component_masks={"rating0": torch.tensor([[True, False]])},
            component_advantages={"rating0": torch.tensor([1.0])},
            component_weights={"rating0": 1.0},
        )
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(ratio.grad).all())

    def test_negative_advantage_reduces_selected_action_logprob(self) -> None:
        log_ratio = torch.zeros((1, 1), requires_grad=True)
        ratio = log_ratio.exp()
        total, _, _ = compute_component_policy_loss(
            ratio=ratio,
            clipped_ratio=ratio,
            rollout_is_weights=torch.ones_like(ratio),
            completion_mask=torch.ones((1, 1), dtype=torch.bool),
            component_masks={"edit_gate": torch.ones((1, 1), dtype=torch.bool)},
            component_advantages={"edit_gate": torch.tensor([-0.7])},
            component_weights={"edit_gate": 1.0},
        )
        total.backward()
        self.assertGreater(float(log_ratio.grad.item()), 0.0)

    def test_component_loss_exposes_only_standard_grpo_inputs(self) -> None:
        parameters = inspect.signature(compute_component_policy_loss).parameters
        self.assertNotIn("sampled_token_logps", parameters)
        self.assertNotIn("negative_unlikelihood_components", parameters)

    def test_sequence_format_penalty_updates_every_selected_token(self) -> None:
        log_ratio = torch.zeros((1, 4), requires_grad=True)
        ratio = log_ratio.exp()
        total, _, _ = compute_component_policy_loss(
            ratio=ratio,
            clipped_ratio=ratio,
            rollout_is_weights=torch.ones_like(ratio),
            completion_mask=torch.ones((1, 4), dtype=torch.bool),
            component_masks={
                "format_a0": torch.tensor([[True, True, True, True]])
            },
            component_advantages={"format_a0": torch.tensor([-1.0])},
            component_weights={"format_a0": 1.0},
        )
        total.backward()
        self.assertTrue(all(value > 0.0 for value in log_ratio.grad[0].tolist()))

    def test_dapo_token_mean_matches_microchunk_accumulation_with_unequal_lengths(self) -> None:
        full_log_ratio = torch.zeros((2, 4), requires_grad=True)
        mask = torch.tensor([[True, False, False, False], [True, True, True, False]])
        advantages = torch.tensor([1.0, -0.5])
        full_ratio = full_log_ratio.exp()
        full_loss, _, _ = compute_component_policy_loss(
            ratio=full_ratio,
            clipped_ratio=full_ratio,
            rollout_is_weights=torch.ones_like(full_ratio),
            completion_mask=mask,
            component_masks={"dapo_policy": mask},
            component_advantages={"dapo_policy": advantages},
            component_weights={"dapo_policy": 1.0},
            component_denominators={"dapo_policy": 4},
            normalization="token_mean",
        )
        full_loss.backward()
        full_gradient = full_log_ratio.grad.detach().clone()

        chunk_gradient = torch.zeros_like(full_gradient)
        chunk_loss_value = 0.0
        for index in range(2):
            chunk_log_ratio = torch.zeros((1, 4), requires_grad=True)
            chunk_ratio = chunk_log_ratio.exp()
            chunk_loss, _, _ = compute_component_policy_loss(
                ratio=chunk_ratio,
                clipped_ratio=chunk_ratio,
                rollout_is_weights=torch.ones_like(chunk_ratio),
                completion_mask=mask[index:index + 1],
                component_masks={"dapo_policy": mask[index:index + 1]},
                component_advantages={"dapo_policy": advantages[index:index + 1]},
                component_weights={"dapo_policy": 1.0},
                component_denominators={"dapo_policy": 4},
                normalization="token_mean",
            )
            chunk_loss.backward()
            chunk_gradient[index] = chunk_log_ratio.grad[0]
            chunk_loss_value += float(chunk_loss.detach())

        self.assertAlmostEqual(float(full_loss.detach()), chunk_loss_value, places=6)
        self.assertTrue(torch.allclose(full_gradient, chunk_gradient, atol=1e-7, rtol=0.0))

    def test_dapo_token_mean_matches_distributed_rank_average(self) -> None:
        masks = [
            torch.tensor([[True, False, False, False]]),
            torch.tensor([[True, True, True, False]]),
        ]
        advantages = [torch.tensor([1.0]), torch.tensor([-0.5])]
        rank_losses = []
        for mask, advantage in zip(masks, advantages):
            ratio = torch.ones((1, 4))
            loss, _, _ = compute_component_policy_loss(
                ratio=ratio,
                clipped_ratio=ratio,
                rollout_is_weights=ratio,
                completion_mask=mask,
                component_masks={"dapo_policy": mask},
                component_advantages={"dapo_policy": advantage},
                component_weights={"dapo_policy": 1.0},
                component_denominators={"dapo_policy": 4 / 2},
                normalization="token_mean",
            )
            rank_losses.append(loss)

        distributed_average = torch.stack(rank_losses).mean()
        full_mask = torch.cat(masks, dim=0)
        full_ratio = torch.ones((2, 4))
        full_loss, _, _ = compute_component_policy_loss(
            ratio=full_ratio,
            clipped_ratio=full_ratio,
            rollout_is_weights=full_ratio,
            completion_mask=full_mask,
            component_masks={"dapo_policy": full_mask},
            component_advantages={"dapo_policy": torch.cat(advantages)},
            component_weights={"dapo_policy": 1.0},
            component_denominators={"dapo_policy": 4},
            normalization="token_mean",
        )
        self.assertAlmostEqual(float(distributed_average), float(full_loss), places=6)

    def test_zero_mask_shape_padding_matches_training_only_effective_rows(self) -> None:
        active_mask = torch.tensor(
            [[True, False, False, False], [True, True, True, False]]
        )
        active_advantages = torch.tensor([1.0, -0.5])
        active_log_ratio = torch.zeros((2, 4), requires_grad=True)
        active_ratio = active_log_ratio.exp()
        active_loss, _, _ = compute_component_policy_loss(
            ratio=active_ratio,
            clipped_ratio=active_ratio,
            rollout_is_weights=torch.ones_like(active_ratio),
            completion_mask=active_mask,
            component_masks={"dapo_policy": active_mask},
            component_advantages={"dapo_policy": active_advantages},
            component_weights={"dapo_policy": 1.0},
            component_denominators={"dapo_policy": 4},
            normalization="token_mean",
        )
        active_loss.backward()

        padded_mask = torch.cat(
            [active_mask, torch.zeros((2, 4), dtype=torch.bool)], dim=0
        )
        padded_log_ratio = torch.zeros((4, 4), requires_grad=True)
        padded_ratio = padded_log_ratio.exp()
        padded_loss, _, _ = compute_component_policy_loss(
            ratio=padded_ratio,
            clipped_ratio=padded_ratio,
            rollout_is_weights=torch.ones_like(padded_ratio),
            completion_mask=padded_mask,
            component_masks={"dapo_policy": padded_mask},
            component_advantages={
                "dapo_policy": torch.tensor([1.0, -0.5, 0.0, 0.0])
            },
            component_weights={"dapo_policy": 1.0},
            component_denominators={"dapo_policy": 4},
            normalization="token_mean",
        )
        padded_loss.backward()

        self.assertAlmostEqual(float(active_loss), float(padded_loss), places=6)
        self.assertTrue(
            torch.allclose(
                active_log_ratio.grad,
                padded_log_ratio.grad[:2],
                atol=1e-7,
                rtol=0.0,
            )
        )
        self.assertTrue(torch.equal(padded_log_ratio.grad[2:], torch.zeros((2, 4))))

    def test_component_kl_uses_an_independent_sequence_mean_per_segment(self) -> None:
        per_token_kl = torch.tensor(
            [[0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8]],
            requires_grad=True,
        )
        completion_mask = torch.ones((1, 7), dtype=torch.bool)
        total, components, active_count = compute_component_kl_loss(
            per_token_kl=per_token_kl,
            completion_mask=completion_mask,
            component_masks={
                "reasoning": torch.tensor(
                    [[True, True, True, True, True, True, False]]
                ),
                "rating0": torch.tensor(
                    [[False, False, False, False, False, False, True]]
                ),
            },
            component_betas={"reasoning": 0.01, "rating0": 0.02},
        )
        self.assertEqual(active_count, 1)
        self.assertAlmostEqual(float(components["reasoning"]), 0.004, places=7)
        self.assertAlmostEqual(float(components["rating0"]), 0.016, places=7)
        self.assertAlmostEqual(float(total), 0.020, places=7)

    def test_component_kl_has_zero_gradient_outside_selected_segments(self) -> None:
        per_token_kl = torch.tensor(
            [[0.2, 0.3, 0.4, 0.5, 0.6]],
            requires_grad=True,
        )
        total, _, _ = compute_component_kl_loss(
            per_token_kl=per_token_kl,
            completion_mask=torch.ones((1, 5), dtype=torch.bool),
            component_masks={
                "reasoning": torch.tensor([[False, True, True, False, False]]),
                "rating0": torch.tensor([[False, False, False, False, True]]),
            },
            component_betas={"reasoning": 0.01, "rating0": 0.02},
        )
        total.backward()
        self.assertEqual(float(per_token_kl.grad[0, 0]), 0.0)
        self.assertEqual(float(per_token_kl.grad[0, 3]), 0.0)
        self.assertAlmostEqual(float(per_token_kl.grad[0, 1]), 0.005, places=7)
        self.assertAlmostEqual(float(per_token_kl.grad[0, 2]), 0.005, places=7)
        self.assertAlmostEqual(float(per_token_kl.grad[0, 4]), 0.02, places=7)

    def test_sampled_k3_kl_updates_only_the_selected_logprob_segments(self) -> None:
        current_logps = torch.zeros((1, 6), requires_grad=True)
        ref_logps = torch.tensor([[0.2, 0.1, -0.1, 0.3, -0.2, 0.4]])
        ref_minus_current = ref_logps - current_logps
        sampled_k3 = (
            torch.exp(ref_minus_current)
            - ref_minus_current
            - 1.0
        )
        total, _, _ = compute_component_kl_loss(
            per_token_kl=sampled_k3,
            completion_mask=torch.ones((1, 6), dtype=torch.bool),
            component_masks={
                "reasoning": torch.tensor(
                    [[False, True, True, False, False, False]]
                ),
                "rating0": torch.tensor(
                    [[False, False, False, False, False, True]]
                ),
            },
            component_betas={"reasoning": 0.01, "rating0": 0.02},
        )
        total.backward()

        for index in (0, 3, 4):
            self.assertEqual(float(current_logps.grad[0, index]), 0.0)
        for index in (1, 2, 5):
            self.assertNotEqual(float(current_logps.grad[0, index]), 0.0)

    def test_component_kl_microchunks_match_full_batch(self) -> None:
        completion_mask = torch.ones((2, 4), dtype=torch.bool)
        masks = {
            "reasoning": torch.tensor(
                [[True, True, False, False], [True, True, True, False]]
            ),
            "rating0": torch.tensor(
                [[False, False, False, True], [False, False, False, True]]
            ),
        }
        values = torch.tensor(
            [[0.2, 0.4, 9.0, 0.6], [0.1, 0.3, 0.5, 0.7]],
            requires_grad=True,
        )
        full_loss, _, _ = compute_component_kl_loss(
            per_token_kl=values,
            completion_mask=completion_mask,
            component_masks=masks,
            component_betas={"reasoning": 0.01, "rating0": 0.02},
            component_denominators={"reasoning": 2, "rating0": 2},
        )
        full_loss.backward()
        full_gradient = values.grad.detach().clone()

        chunk_gradient = torch.zeros_like(full_gradient)
        chunk_loss_value = 0.0
        for index in range(2):
            chunk_values = values.detach()[index:index + 1].clone().requires_grad_(True)
            chunk_loss, _, _ = compute_component_kl_loss(
                per_token_kl=chunk_values,
                completion_mask=completion_mask[index:index + 1],
                component_masks={
                    name: mask[index:index + 1] for name, mask in masks.items()
                },
                component_betas={"reasoning": 0.01, "rating0": 0.02},
                component_denominators={"reasoning": 2, "rating0": 2},
            )
            chunk_loss.backward()
            chunk_gradient[index] = chunk_values.grad[0]
            chunk_loss_value += float(chunk_loss.detach())

        self.assertAlmostEqual(float(full_loss.detach()), chunk_loss_value, places=7)
        self.assertTrue(torch.allclose(full_gradient, chunk_gradient, atol=1e-8, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
