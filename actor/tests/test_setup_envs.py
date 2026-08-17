from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "setup_envs.sh"
FAKE_CONDA = "/opt/conda/bin/conda"


class SetupEnvironmentsTests(unittest.TestCase):
    def run_setup(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT), "--conda", FAKE_CONDA, *arguments],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_inference_dry_run_uses_both_pinned_gpu_environments(self) -> None:
        result = self.run_setup("--profile", "inference", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("environment/actor-judge.yml", result.stdout)
        self.assertIn("requirements/actor-judge.txt", result.stdout)
        self.assertIn("environment/editor.yml", result.stdout)
        self.assertIn("requirements/editor.txt", result.stdout)
        self.assertIn("swift", result.stdout)
        self.assertIn("vllm", result.stdout)
        self.assertIn("diffusers", result.stdout)
        self.assertNotIn("FlashAttention wheel", result.stderr)

    def test_training_dry_run_installs_explicit_flash_attention_wheel(self) -> None:
        result = self.run_setup(
            "--profile",
            "training",
            "--flash-attn-wheel",
            "/tmp/validated_flash_attn.whl",
            "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("/tmp/validated_flash_attn.whl", result.stdout)
        self.assertIn("flash_attn", result.stdout)
        self.assertIn(".env.example", result.stdout)

    def test_test_profile_uses_cpu_torch_and_runs_release_suite(self) -> None:
        result = self.run_setup("--profile", "test", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("environment/test.yml", result.stdout)
        self.assertIn("https://download.pytorch.org/whl/cpu", result.stdout)
        self.assertIn("scripts/test_release.sh", result.stdout)

    def test_rejects_unknown_profile(self) -> None:
        result = self.run_setup("--profile", "unknown", "--dry-run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported profile", result.stderr)


if __name__ == "__main__":
    unittest.main()
