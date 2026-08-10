from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


JUDGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JUDGE_ROOT))

from release_manifest import (  # noqa: E402
    LEGACY_SCHEMA,
    REQUIRED_RELEASE_FILES as PORTABLE_REQUIRED_FILES,
    PORTABLE_SCHEMA,
    full_tree_sha256 as sha256_tree,
    selected_export_tree_sha256 as export_tree_digest,
    validate_checkpoint_manifest,
)


ASSET_ID = "source-e5-judge-step725"
HUB_SUBFOLDER = "judge/source-e5"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class JudgeReleaseManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "download" / "judge" / "source-e5"
        self.model.mkdir(parents=True)
        for index, name in enumerate(PORTABLE_REQUIRED_FILES, start=1):
            (self.model / name).write_bytes(
                (f"synthetic:{index}:{name}\n".encode("utf-8")) * index
            )
        self.source_tree = "a" * 64
        self.portable = self._portable_manifest()
        self.manifest = self.model / "provenance.json"
        self._write(self.portable)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _file_records(self, model: Path | None = None) -> list[dict[str, object]]:
        root = model or self.model
        return [
            {
                "name": name,
                "size_bytes": (root / name).stat().st_size,
                "sha256": sha256_file(root / name),
            }
            for name in PORTABLE_REQUIRED_FILES
        ]

    def _portable_manifest(self) -> dict[str, object]:
        records = self._file_records()
        return {
            "schema_version": PORTABLE_SCHEMA,
            "asset_id": ASSET_ID,
            "source_logical_id": ASSET_ID,
            "hub_subfolder": HUB_SUBFOLDER,
            "source_identity": {
                "source_checkpoint_tree_sha256_recorded": self.source_tree,
                "selected_export_tree_sha256": export_tree_digest(records),
                "file_count": len(records),
                "total_bytes": sum(int(record["size_bytes"]) for record in records),
                "ignored_source_entries": [],
            },
            "files": records,
        }

    def _write(self, payload: dict[str, object], path: Path | None = None) -> Path:
        output = path or self.manifest
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return output

    @staticmethod
    def _identity(payload: dict[str, object]) -> dict[str, object]:
        identity = payload["source_identity"]
        assert isinstance(identity, dict)
        return identity

    def _validate_portable(
        self,
        *,
        payload: dict[str, object] | None = None,
        model: Path | None = None,
        manifest: Path | None = None,
        asset_id: str = ASSET_ID,
        hub_subfolder: str = HUB_SUBFOLDER,
        source_tree: str | None = None,
        export_tree: str | None = None,
        expected_file_count: int = 10,
        verify_files: bool = True,
    ) -> dict[str, object]:
        value = payload or self.portable
        if payload is not None:
            self._write(payload, manifest)
        identity = self._identity(value)
        return validate_checkpoint_manifest(
            manifest_path=manifest or self.manifest,
            model_path=model or self.model,
            expected_asset_id=asset_id,
            expected_hub_subfolder=hub_subfolder,
            expected_source_tree=source_tree or self.source_tree,
            expected_export_tree=export_tree
            or str(self._identity(self.portable)["selected_export_tree_sha256"]),
            expected_file_count=expected_file_count,
            expected_total_bytes=int(self._identity(self.portable)["total_bytes"]),
            verify_files=verify_files,
        )

    def assertPortableRejected(self, payload: dict[str, object]) -> None:  # noqa: N802
        try:
            with self.assertRaises(ValueError):
                self._validate_portable(payload=payload)
        finally:
            self._write(self.portable)

    def test_portable_manifest_verifies_exact_ten_file_export(self) -> None:
        result = self._validate_portable()
        identity = self._identity(self.portable)
        self.assertEqual(result["manifest_kind"], "portable_hub_export")
        self.assertEqual(result["schema_version"], PORTABLE_SCHEMA)
        self.assertEqual(result["asset_id"], ASSET_ID)
        self.assertEqual(result["file_count"], 10)
        self.assertTrue(result["files_verified"])
        self.assertEqual(
            result["selected_export_tree_sha256"],
            identity["selected_export_tree_sha256"],
        )

    def test_selected_tree_algorithm_matches_hub_exporter(self) -> None:
        exporter_path = JUDGE_ROOT.parent / "scripts" / "hf_export_checkpoints.py"
        spec = importlib.util.spec_from_file_location("mriqa2_hf_exporter_test", exporter_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        records = self._file_records()
        self.assertEqual(export_tree_digest(records), module._tree_digest(records))

    def test_portable_manifest_is_relocatable(self) -> None:
        relocated = self.root / "another-machine" / "models" / "judge"
        relocated.parent.mkdir(parents=True)
        shutil.move(str(self.model), relocated)
        result = self._validate_portable(
            model=relocated,
            manifest=relocated / "provenance.json",
        )
        self.assertEqual(result["manifest_kind"], "portable_hub_export")

    def test_portable_rejects_asset_logical_id_folder_and_source_identity_tampering(self) -> None:
        mutations: list[dict[str, object]] = []
        for key, value in (
            ("asset_id", "another-judge"),
            ("source_logical_id", "another-source"),
            ("hub_subfolder", "judge/other"),
        ):
            payload = copy.deepcopy(self.portable)
            payload[key] = value
            mutations.append(payload)
        payload = copy.deepcopy(self.portable)
        self._identity(payload)["source_checkpoint_tree_sha256_recorded"] = "b" * 64
        mutations.append(payload)
        for payload in mutations:
            with self.subTest(payload=payload):
                self.assertPortableRejected(payload)

        with self.assertRaises(ValueError):
            self._validate_portable(asset_id="another-judge")
        with self.assertRaises(ValueError):
            self._validate_portable(hub_subfolder="judge/other")
        with self.assertRaises(ValueError):
            self._validate_portable(source_tree="b" * 64)

    def test_portable_rejects_selected_and_independent_export_tree_tampering(self) -> None:
        payload = copy.deepcopy(self.portable)
        self._identity(payload)["selected_export_tree_sha256"] = "b" * 64
        self.assertPortableRejected(payload)
        with self.assertRaises(ValueError):
            self._validate_portable(export_tree="c" * 64)

    def test_portable_rejects_file_name_size_hash_and_count_tampering(self) -> None:
        mutations: list[dict[str, object]] = []

        renamed = copy.deepcopy(self.portable)
        records = renamed["files"]
        assert isinstance(records, list) and isinstance(records[0], dict)
        records[0]["name"] = "renamed.safetensors"
        mutations.append(renamed)

        duplicated = copy.deepcopy(self.portable)
        records = duplicated["files"]
        assert isinstance(records, list) and isinstance(records[0], dict)
        assert isinstance(records[1], dict)
        records[1]["name"] = records[0]["name"]
        mutations.append(duplicated)

        resized = copy.deepcopy(self.portable)
        records = resized["files"]
        assert isinstance(records, list) and isinstance(records[0], dict)
        records[0]["size_bytes"] = int(records[0]["size_bytes"]) + 1
        mutations.append(resized)

        boolean_size = copy.deepcopy(self.portable)
        records = boolean_size["files"]
        assert isinstance(records, list) and isinstance(records[0], dict)
        records[0]["size_bytes"] = True
        mutations.append(boolean_size)

        rehashed = copy.deepcopy(self.portable)
        records = rehashed["files"]
        assert isinstance(records, list) and isinstance(records[0], dict)
        records[0]["sha256"] = "d" * 64
        mutations.append(rehashed)

        missing = copy.deepcopy(self.portable)
        records = missing["files"]
        assert isinstance(records, list)
        records.pop()
        mutations.append(missing)

        extra = copy.deepcopy(self.portable)
        records = extra["files"]
        assert isinstance(records, list)
        records.append({"name": "optimizer.pt", "size_bytes": 1, "sha256": "e" * 64})
        mutations.append(extra)

        malformed = copy.deepcopy(self.portable)
        records = malformed["files"]
        assert isinstance(records, list)
        records[0] = "not-an-object"
        mutations.append(malformed)

        for payload in mutations:
            with self.subTest(payload=payload):
                self.assertPortableRejected(payload)

    def test_portable_rejects_identity_count_total_and_caller_contract_tampering(self) -> None:
        for key, value in (("file_count", 9), ("total_bytes", 1)):
            payload = copy.deepcopy(self.portable)
            self._identity(payload)[key] = value
            with self.subTest(key=key):
                self.assertPortableRejected(payload)
        with self.assertRaises(ValueError):
            self._validate_portable(expected_file_count=9)

    def test_portable_rejects_disk_content_missing_file_and_symlink_tampering(self) -> None:
        target = self.model / PORTABLE_REQUIRED_FILES[0]
        original = target.read_bytes()
        target.write_bytes(original + b"tampered")
        with self.assertRaises(ValueError):
            self._validate_portable()
        target.write_bytes(original)

        target = self.model / PORTABLE_REQUIRED_FILES[-1]
        original = target.read_bytes()
        target.unlink()
        with self.assertRaises(ValueError):
            self._validate_portable()

        target.write_bytes(original)
        external = self.root / "outside.bin"
        external.write_bytes(original)
        target.unlink()
        target.symlink_to(external)
        with self.assertRaises(ValueError):
            self._validate_portable()

    def _legacy_fixture(self) -> tuple[Path, Path, str]:
        run = self.root / "legacy-run"
        model = run / "train" / "checkpoint-725"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
        (model / "preprocessor_config.json").write_text("{}\n", encoding="utf-8")
        for index, name in enumerate(PORTABLE_REQUIRED_FILES[:2], start=1):
            (model / name).write_bytes(bytes([index]) * index)
        tree = sha256_tree(model)
        payload: dict[str, object] = {
            "schema_version": LEGACY_SCHEMA,
            "run_dir": str(run),
            "checkpoint": str(model),
            "status": "promoted",
            "usable": True,
            "checkpoint_identity": {
                "tree_sha256": tree,
                "artifact_validation": {
                    "valid": True,
                    "errors": [],
                    "weight_files": list(PORTABLE_REQUIRED_FILES[:2]),
                    "total_weight_bytes": 3,
                },
            },
        }
        path = self._write(payload, self.root / "legacy.json")
        return path, model, tree

    def test_legacy_manifest_remains_supported_with_full_tree_resolution(self) -> None:
        path, model, tree = self._legacy_fixture()
        result = validate_checkpoint_manifest(
            manifest_path=path,
            model_path=model,
            expected_asset_id=ASSET_ID,
            expected_hub_subfolder=HUB_SUBFOLDER,
            expected_source_tree=tree,
            expected_export_tree="b" * 64,
            expected_file_count=10,
            expected_total_bytes=1,
            verify_files=True,
            expected_legacy_weight_bytes=3,
        )
        self.assertEqual(result["manifest_kind"], "promoted_training_checkpoint")

        (model / "config.json").write_text('{"tampered":true}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_checkpoint_manifest(
                manifest_path=path,
                model_path=model,
                expected_asset_id=ASSET_ID,
                expected_hub_subfolder=HUB_SUBFOLDER,
                expected_source_tree=tree,
                expected_export_tree="b" * 64,
                expected_file_count=10,
                expected_total_bytes=1,
                verify_files=True,
                expected_legacy_weight_bytes=3,
            )

    def test_legacy_manifest_rejects_metadata_and_path_tampering(self) -> None:
        path, model, tree = self._legacy_fixture()
        base = json.loads(path.read_text(encoding="utf-8"))
        mutations: list[dict[str, object]] = []
        for key, value in (("status", "quarantined"), ("usable", False)):
            payload = copy.deepcopy(base)
            payload[key] = value
            mutations.append(payload)
        payload = copy.deepcopy(base)
        identity = payload["checkpoint_identity"]
        assert isinstance(identity, dict)
        identity["tree_sha256"] = "c" * 64
        mutations.append(payload)
        payload = copy.deepcopy(base)
        identity = payload["checkpoint_identity"]
        assert isinstance(identity, dict)
        artifacts = identity["artifact_validation"]
        assert isinstance(artifacts, dict)
        artifacts["total_weight_bytes"] = 1
        mutations.append(payload)

        for index, payload in enumerate(mutations):
            mutated = self._write(payload, self.root / f"legacy-mutated-{index}.json")
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_checkpoint_manifest(
                    manifest_path=mutated,
                    model_path=model,
                    expected_asset_id=ASSET_ID,
                    expected_hub_subfolder=HUB_SUBFOLDER,
                    expected_source_tree=tree,
                    expected_export_tree="b" * 64,
                    expected_file_count=10,
                    expected_total_bytes=1,
                    verify_files=True,
                    expected_legacy_weight_bytes=3,
                )

        relocated = self.root / "relocated-legacy"
        shutil.move(str(model), relocated)
        with self.assertRaises(ValueError):
            validate_checkpoint_manifest(
                manifest_path=path,
                model_path=relocated,
                expected_asset_id=ASSET_ID,
                expected_hub_subfolder=HUB_SUBFOLDER,
                expected_source_tree=tree,
                expected_export_tree="b" * 64,
                expected_file_count=10,
                expected_total_bytes=1,
                verify_files=True,
                expected_legacy_weight_bytes=3,
            )

    def test_cli_validates_portable_and_fails_closed_after_tampering(self) -> None:
        identity = self._identity(self.portable)
        command = [
            sys.executable,
            str(JUDGE_ROOT / "release_manifest.py"),
            "--manifest",
            str(self.manifest),
            "--model-path",
            str(self.model),
            "--expected-asset-id",
            ASSET_ID,
            "--expected-hub-subfolder",
            HUB_SUBFOLDER,
            "--expected-source-tree",
            self.source_tree,
            "--expected-export-tree",
            str(identity["selected_export_tree_sha256"]),
            "--expected-file-count",
            "10",
            "--expected-total-bytes",
            str(identity["total_bytes"]),
            "--verify-files",
            "1",
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        result = json.loads(completed.stdout)
        self.assertEqual(result["manifest_kind"], "portable_hub_export")
        self.assertTrue(result["files_verified"])

        (self.model / PORTABLE_REQUIRED_FILES[3]).write_bytes(b"tampered")
        failed = subprocess.run(command, check=False, capture_output=True, text=True)
        self.assertEqual(failed.returncode, 2)
        self.assertIn("mismatch", failed.stderr)


if __name__ == "__main__":
    unittest.main()
