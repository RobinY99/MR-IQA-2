from __future__ import annotations

import copy
import contextlib
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPORTER = load_module("mriqa2_hf_exporter_tests", ROOT / "scripts" / "hf_export_checkpoints.py")
UPLOADER = load_module("mriqa2_hf_uploader_tests", ROOT / "scripts" / "hf_upload_release.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HuggingFaceReleaseToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = json.loads(
            (ROOT / "checkpoints" / "hf_checkpoint_manifest.json").read_text(encoding="utf-8")
        )
        self.sources: dict[str, Path] = {}
        for index, asset in enumerate(self.manifest["assets"], start=1):
            source = self.root / "sources" / asset["role"]
            source.mkdir(parents=True)
            self.sources[asset["source_logical_id"]] = source
            records = []
            verified = {}
            for name in asset["required_files"]:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"fixture-{index}:{name}\n".encode("utf-8"))
                record = {"name": name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
                records.append(record)
                verified[name] = {key: record[key] for key in ("size_bytes", "sha256")}
            asset["verified_files"] = verified
            asset["expected_file_count"] = len(records)
            asset["expected_total_bytes"] = sum(record["size_bytes"] for record in records)
            asset["expected_export_tree_sha256"] = EXPORTER._tree_digest(records)
            if asset["role"] in {"actor", "judge"}:
                asset["source_checkpoint_tree_sha256"] = str(index) * 64

        editor_source = self.sources["flux2-klein-4b-editor"]
        license_path = editor_source / "LICENSE.md"
        license_path.write_text("Apache License 2.0 fixture\n", encoding="utf-8")
        self.manifest["root_license"]["size_bytes"] = license_path.stat().st_size
        self.manifest["root_license"]["sha256"] = sha256(license_path)
        self.manifest["expected_runtime_file_count"] = sum(
            asset["expected_file_count"] for asset in self.manifest["assets"]
        )
        self.manifest["expected_checkpoint_bytes"] = sum(
            asset["expected_total_bytes"] for asset in self.manifest["assets"]
        )
        self.template = self.root / "template"
        self.template.mkdir()
        (self.template / "README.md").write_text("# Test model\n", encoding="utf-8")
        (self.template / ".gitattributes").write_text(
            "*.safetensors filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8"
        )
        self.static_source = (
            self.template / "assets" / "actor_editor" / "sample_0001" / "source.png"
        )
        self.static_edited = self.static_source.with_name("edited.png")
        self.static_sample = (
            self.template / "examples" / "actor_editor" / "sample_0001.json"
        )
        self.static_source.parent.mkdir(parents=True)
        self.static_sample.parent.mkdir(parents=True)
        self.static_source.write_bytes(EXPORTER.PNG_SIGNATURE + b"source-fixture")
        self.static_edited.write_bytes(EXPORTER.PNG_SIGNATURE + b"edited-fixture")
        self.static_sample.write_text(
            '{"sample_id":"sample_0001"}\n', encoding="utf-8"
        )
        self.output = self.root / "staging"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def materialize(self) -> None:
        assets = EXPORTER.validate_manifest(self.manifest, allow_test_payloads=True)
        inspections = [
            EXPORTER.inspect_source(self.sources[asset["source_logical_id"]], asset)
            for asset in assets
        ]
        source_map = {key: str(value) for key, value in self.sources.items()}
        license_inspection = EXPORTER.inspect_root_license(source_map, self.manifest)
        EXPORTER.materialize(
            template=self.template,
            output=self.output,
            manifest=self.manifest,
            inspections=inspections,
            license_inspection=license_inspection,
            transfer_mode="copy",
        )

    def test_production_manifest_has_exact_three_model_contract(self) -> None:
        production = json.loads(
            (ROOT / "checkpoints" / "hf_checkpoint_manifest.json").read_text(encoding="utf-8")
        )
        assets = EXPORTER.validate_manifest(production)
        self.assertEqual([asset["role"] for asset in assets], ["actor", "judge", "editor"])
        self.assertEqual([asset["hub_subfolder"] for asset in assets], ["actor", "judge", "editor"])
        self.assertEqual([len(asset["required_files"]) for asset in assets], [10, 10, 18])
        self.assertEqual(production["root_license"]["published_path"], "LICENSE")

    def test_materializer_and_uploader_accept_only_the_exact_public_tree(self) -> None:
        self.materialize()
        result = UPLOADER.validate_release(
            self.output,
            self.manifest,
            allow_test_payloads=True,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["asset_count"], 3)
        self.assertEqual(result["runtime_file_count"], 38)
        self.assertEqual(result["public_file_count"], 46)
        self.assertEqual(
            {item.name for item in self.output.iterdir()},
            {
                ".gitattributes",
                "README.md",
                "LICENSE",
                "actor",
                "judge",
                "editor",
                "assets",
                "examples",
            },
        )
        self.assertFalse((self.output / "checkpoint_manifest.json").exists())
        self.assertFalse((self.output / "SHA256SUMS.full").exists())
        self.assertFalse((self.output / "export_report.json").exists())

    def test_actor_editor_static_paths_are_materialized_and_accepted(self) -> None:
        self.materialize()
        result = UPLOADER.validate_release(
            self.output,
            self.manifest,
            allow_test_payloads=True,
        )

        self.assertEqual(result["public_file_count"], 46)
        self.assertEqual(
            (self.output / self.static_source.relative_to(self.template)).read_bytes(),
            EXPORTER.PNG_SIGNATURE + b"source-fixture",
        )
        self.assertEqual(
            (self.output / self.static_edited.relative_to(self.template)).read_bytes(),
            EXPORTER.PNG_SIGNATURE + b"edited-fixture",
        )
        self.assertEqual(
            (self.output / self.static_sample.relative_to(self.template)).read_text(
                encoding="utf-8"
            ),
            '{"sample_id":"sample_0001"}\n',
        )

    def test_uploader_rejects_paths_outside_approved_static_prefixes(self) -> None:
        self.materialize()
        unapproved = self.output / "examples" / "actor_editor" / "credentials.env"
        unapproved.parent.mkdir(parents=True, exist_ok=True)
        unapproved.write_text("HF_TOKEN=hf_not_a_real_token_but_must_be_rejected\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "extra"):
            UPLOADER.validate_release(
                self.output,
                self.manifest,
                allow_test_payloads=True,
            )

    def test_remote_replacement_deletes_every_path_absent_from_verified_tree(self) -> None:
        remote = {
            "README.md",
            "actor/config.json",
            "assets/actor_editor/sample_0001/source.png",
            "examples/actor_editor/sample_0001.json",
            "obsolete.bin",
        }
        local = {"README.md", "actor/config.json"}

        self.assertEqual(
            UPLOADER._remote_delete_paths(remote, local),
            [
                "assets/actor_editor/sample_0001/source.png",
                "examples/actor_editor/sample_0001.json",
                "obsolete.bin",
            ],
        )

    def test_uploader_rejects_completion_cache_and_any_other_extra_path(self) -> None:
        self.materialize()
        completion = self.output / "actors" / "actor-completion-e5-step1455"
        completion.mkdir(parents=True)
        (completion / "config.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "root entries must be exactly"):
            UPLOADER.validate_release(
                self.output,
                self.manifest,
                allow_test_payloads=True,
            )
        (completion / "config.json").unlink()
        completion.rmdir()
        completion.parent.rmdir()

        cache = self.output / "training_assets"
        cache.mkdir()
        (cache / "original_score_cache.sqlite").write_bytes(b"not published")
        with self.assertRaisesRegex(ValueError, "root entries must be exactly"):
            UPLOADER.validate_release(
                self.output,
                self.manifest,
                allow_test_payloads=True,
            )

    def test_manifest_rejects_recursive_parent_escape_and_editor_identity_changes(self) -> None:
        escaped = copy.deepcopy(self.manifest)
        escaped["assets"][2]["required_files"][0] = "../model_index.json"
        with self.assertRaisesRegex(ValueError, "unsafe relative path"):
            EXPORTER.validate_manifest(escaped)

        wrong_revision = copy.deepcopy(self.manifest)
        wrong_revision["assets"][2]["upstream_revision"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "upstream_revision mismatch"):
            EXPORTER.validate_manifest(wrong_revision)

    def test_malicious_custom_manifest_cannot_replace_exact_model_file_sets(self) -> None:
        malicious = copy.deepcopy(self.manifest)
        total_bytes = 0
        for asset in malicious["assets"]:
            name = "secret.txt" if asset["role"] != "editor" else "payload/secret.txt"
            record = {
                "name": name,
                "size_bytes": 32,
                "sha256": "a" * 64,
            }
            asset["required_files"] = [name]
            asset["verified_files"] = {
                name: {"size_bytes": record["size_bytes"], "sha256": record["sha256"]}
            }
            asset["expected_file_count"] = 1
            asset["expected_total_bytes"] = record["size_bytes"]
            asset["expected_export_tree_sha256"] = EXPORTER._tree_digest([record])
            total_bytes += record["size_bytes"]
        malicious["expected_runtime_file_count"] = 3
        malicious["expected_checkpoint_bytes"] = total_bytes
        with self.assertRaisesRegex(ValueError, "required_files mismatch"):
            EXPORTER.validate_manifest(malicious, allow_test_payloads=True)
        with self.assertRaisesRegex(ValueError, "required_files mismatch"):
            UPLOADER._manifest_assets(malicious, allow_test_payloads=True)

    def test_production_manifest_digest_steps_and_license_are_locked(self) -> None:
        production = json.loads(
            (ROOT / "checkpoints" / "hf_checkpoint_manifest.json").read_text(encoding="utf-8")
        )
        mutations = []
        changed_step = copy.deepcopy(production)
        changed_step["assets"][0]["global_step"] = 1164
        mutations.append(changed_step)
        changed_id = copy.deepcopy(production)
        changed_id["assets"][1]["asset_id"] = "another-judge"
        mutations.append(changed_id)
        changed_license = copy.deepcopy(production)
        changed_license["root_license"]["sha256"] = "b" * 64
        mutations.append(changed_license)
        changed_hash = copy.deepcopy(production)
        changed_hash["assets"][2]["verified_files"]["model_index.json"]["sha256"] = "c" * 64
        mutations.append(changed_hash)
        for payload in mutations:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    EXPORTER.validate_manifest(payload)
                with self.assertRaises(ValueError):
                    UPLOADER._manifest_assets(payload)

    def test_cli_does_not_offer_manifest_or_commit_message_overrides(self) -> None:
        original = list(sys.argv)
        try:
            for option in ("--manifest", "--commit-message"):
                with self.subTest(option=option):
                    sys.argv = ["hf_upload_release.py", "--folder", "stage", option, "unsafe"]
                    with contextlib.redirect_stderr(__import__("io").StringIO()):
                        with self.assertRaises(SystemExit):
                            UPLOADER.parse_args()
        finally:
            sys.argv = original

    def test_uploader_requires_both_mutation_flags(self) -> None:
        original = list(__import__("sys").argv)
        try:
            __import__("sys").argv = [
                "hf_upload_release.py",
                "--folder",
                str(self.root / "absent"),
                "--commit-upload",
            ]
            with self.assertRaisesRegex(ValueError, "requires both"):
                UPLOADER.main()
        finally:
            __import__("sys").argv = original


if __name__ == "__main__":
    unittest.main()
