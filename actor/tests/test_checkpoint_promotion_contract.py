from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "actor" / "scripts"))

from checkpoint_manifest import (  # noqa: E402
    INFERENCE_EXPORT_DIGEST_ALGORITHM,
    INFERENCE_EXPORT_FILES,
    PROMOTED,
    TECHNICALLY_VALID,
    build_checkpoint_manifest,
    build_validation_summary,
    inspect_inference_export,
    resolve_promoted_checkpoint,
    transition_to_promoted,
    transition_to_technically_valid,
    write_manifest,
)


def make_checkpoint(root: Path) -> Path:
    checkpoint = root / "run" / "train" / "checkpoint-291"
    checkpoint.mkdir(parents=True)
    for name in INFERENCE_EXPORT_FILES:
        payload = b"{}\n"
        if name.endswith(".safetensors"):
            payload = ("weights:" + name).encode("utf-8")
        elif name == "model.safetensors.index.json":
            payload = json.dumps(
                {
                    "weight_map": {
                        "layer.0": "model-00001-of-00002.safetensors",
                        "layer.1": "model-00002-of-00002.safetensors",
                    }
                }
            ).encode("utf-8")
        elif name == "chat_template.jinja":
            payload = b"{{ messages }}\n"
        (checkpoint / name).write_bytes(payload)
    # Resume-only and mutable entries are intentionally outside the identity.
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer-v1")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 291}), encoding="utf-8"
    )
    (checkpoint / "rng_state_0.pth").write_bytes(b"rng-v1")
    (checkpoint / ".cache").mkdir()
    (checkpoint / ".cache" / "temporary.bin").write_bytes(b"temporary-v1")
    return checkpoint


def complete_validation_summary(digest: str) -> dict[str, object]:
    return {
        "num_total": 200,
        "num_shards": 8,
        "num_missing_or_bad_gold": 0,
        "batch_generate_exception_count": 0,
        "singleton_generate_exception_count": 0,
        "actor_schema": "reasoning_evidence_solution_rating",
        "actor_result_rows": 200,
        "editor_status": "complete",
        "editor_total_actor_rows": 200,
        "editor_total_rows": 200,
        "editor_service_error_rows": 0,
        "editor_barrier_passed": True,
        "judge_status": "complete",
        "judge_total_actor_rows": 200,
        "judge_total_rows": 200,
        "judge_service_error_rows": 0,
        "all_edits_finished_before_any_judge_request": True,
        "checkpoint_export_tree_sha256": digest,
        "checkpoint_digest_semantics": "selected_inference_export",
        "checkpoint_digest_algorithm": INFERENCE_EXPORT_DIGEST_ALGORITHM,
    }


class CheckpointPromotionContractTest(unittest.TestCase):
    def test_identity_ignores_resume_and_cache_state_but_detects_model_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = make_checkpoint(Path(temporary))
            before = inspect_inference_export(checkpoint)
            self.assertEqual(before["file_count"], 10)
            self.assertEqual(
                [item["name"] for item in before["files"]],
                list(INFERENCE_EXPORT_FILES),
            )

            (checkpoint / "optimizer.pt").write_bytes(b"optimizer-v2")
            (checkpoint / ".cache" / "temporary.bin").write_bytes(b"temporary-v2")
            self.assertEqual(inspect_inference_export(checkpoint), before)

            (checkpoint / "model-00001-of-00002.safetensors").write_bytes(
                b"changed weights"
            )
            self.assertNotEqual(
                inspect_inference_export(checkpoint)["selected_export_tree_sha256"],
                before["selected_export_tree_sha256"],
            )

    def test_quarantine_technical_validation_promotion_and_promoted_only_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_checkpoint(root)
            manifest_path = root / "state" / "epoch1.json"
            manifest = build_checkpoint_manifest(
                root / "run",
                checkpoint,
                run_id="synthetic-epoch1",
                parent_checkpoint=root / "base",
                provenance={
                    "data_sha256": "a" * 64,
                    "prompt_hash": "b" * 64,
                    "code_sha256": "c" * 64,
                },
            )
            self.assertEqual(manifest["status"], "quarantined")
            write_manifest(manifest_path, manifest)
            with self.assertRaisesRegex(RuntimeError, "not promoted"):
                resolve_promoted_checkpoint(manifest_path)

            technical = transition_to_technically_valid(
                manifest,
                trainer_exit_code=0,
                trajectory_summary={
                    "num_rows": 144,
                    "num_rank_shards": 4,
                    "unique_trajectory_ids": 144,
                    "credit_integrity_rate": 1.0,
                    "non_finite_count": 0,
                },
                wandb_url="offline://synthetic/epoch1",
            )
            self.assertEqual(technical["status"], TECHNICALLY_VALID)
            digest = technical["checkpoint_identity"][
                "selected_export_tree_sha256"
            ]
            promoted = transition_to_promoted(
                technical,
                validation_summary=complete_validation_summary(digest),
                approval="synthetic-validation200",
                thresholds={"num_shards": 8},
                validation_policy="comparison_observational",
            )
            self.assertEqual(promoted["status"], PROMOTED)
            write_manifest(manifest_path, promoted)
            self.assertEqual(resolve_promoted_checkpoint(manifest_path), checkpoint.resolve())

            # Mutable optimizer state is not an inference digest input.
            (checkpoint / "optimizer.pt").write_bytes(b"optimizer-v3")
            self.assertEqual(resolve_promoted_checkpoint(manifest_path), checkpoint.resolve())
            (checkpoint / "model-00002-of-00002.safetensors").write_bytes(
                b"tampered"
            )
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                resolve_promoted_checkpoint(manifest_path)

    def test_validation_summary_requires_complete_200_row_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            digest = "d" * 64
            (root / "actor_outputs" / "validation").mkdir(parents=True)
            (root / "editor_judge").mkdir()
            (root / "state").mkdir()
            (root / "actor_outputs" / "validation" / "merged.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "num_total": 200,
                            "num_shards": 8,
                            "num_missing_or_bad_gold": 0,
                            "batch_generate_exception_count": 0,
                            "singleton_generate_exception_count": 0,
                            "actor_schema": "reasoning_evidence_solution_rating",
                        },
                        "results": [{"index": index} for index in range(200)],
                    }
                ),
                encoding="utf-8",
            )
            (root / "editor_judge" / "editor_summary.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "total_actor_rows": 200,
                        "total_editor_rows": 200,
                        "service_error_rows": 0,
                    }
                ),
                encoding="utf-8",
            )
            (root / "editor_judge" / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "total_actor_rows": 200,
                        "total_judge_rows": 200,
                        "service_error_rows": 0,
                        "all_edits_finished_before_any_judge_request": True,
                    }
                ),
                encoding="utf-8",
            )
            (root / "state" / "editor_barrier.json").write_text(
                json.dumps({"status": "passed"}), encoding="utf-8"
            )
            (root / "contract.json").write_text(
                json.dumps(
                    {
                        "schema_version": "mr_iqa_evaluation_contract_v1",
                        "checkpoint_digest": {
                            "semantics": "selected_inference_export",
                            "algorithm": "synthetic",
                            "sha256": digest,
                            "file_count": 10,
                        },
                    }
                ),
                encoding="utf-8",
            )
            summary = build_validation_summary(root)
            self.assertEqual(summary["actor_result_rows"], 200)
            self.assertEqual(summary["checkpoint_export_tree_sha256"], digest)
            self.assertTrue(summary["editor_barrier_passed"])

    def test_public_shell_chain_uses_promoted_manifest_and_actual_export_digest(self) -> None:
        train = (ROOT / "scripts" / "train.sh").read_text(encoding="utf-8")
        evaluate = (ROOT / "scripts" / "evaluate.sh").read_text(encoding="utf-8")
        self.assertIn('PARENT_MANIFEST="${parent_manifest}"', train)
        self.assertIn('resolve --manifest "${manifest}"', train)
        self.assertIn('ACTOR_MODEL_EXPORT_TREE_SHA256="${checkpoint_digest}"', train)
        self.assertNotIn('"${ACTOR_MODEL_TREE_SHA256:-}"', evaluate)
        self.assertIn('"semantics": "selected_inference_export"', evaluate)


if __name__ == "__main__":
    unittest.main()
