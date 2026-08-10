from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from examples.actor_to_editor import (
    ActorOutputError,
    EditorServiceError,
    atomic_write_json,
    build_editor_request,
    parse_args,
    parse_valid_actor_output,
    run_example,
    validate_editor_response,
)


class ActorToEditorExampleTests(unittest.TestCase):
    def test_editor_receives_solution_verbatim(self) -> None:
        raw = (
            '{"reasoning":{"evidence":"Mild softness is visible on the bicycle frame.",'
            '"solution":"  Apply mild sharpening while preserving the scene.  "},'
            '"rating":"3.25"}'
        )
        payload = parse_valid_actor_output(raw)
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source.png"
            source.write_bytes(b"source")
            request = build_editor_request(
                image_path=source,
                actor_payload=payload,
                seed=123,
                request_index=7,
            )

        self.assertEqual(
            request["positive_prompt"],
            "  Apply mild sharpening while preserving the scene.  ",
        )
        self.assertEqual(request["positive_prompt"], payload["reasoning"]["solution"])
        self.assertEqual(request["negative_prompt"], "")
        self.assertEqual(request["region_prompt"], "")
        self.assertEqual(request["edit_plan"], {})
        self.assertEqual(request["request_index"], 7)
        self.assertEqual(request["seed"], 123)

    def test_invalid_actor_output_is_rejected(self) -> None:
        invalid_outputs = (
            "not JSON",
            '{"reasoning":{"evidence":"blur","solution":"sharpen"},"rating":"5.10"}',
            '{"rating":"3.00","reasoning":{"evidence":"blur","solution":"sharpen"}}',
            '{"reasoning":{"evidence":"blur","solution":""},"rating":"3.00"}',
            '{"reasoning":{"evidence":"blur","solution":"sharpen","extra":1},"rating":"3.00"}',
        )
        for raw in invalid_outputs:
            with self.subTest(raw=raw), self.assertRaises(ActorOutputError):
                parse_valid_actor_output(raw)

        numeric_rating = parse_valid_actor_output(
            '{"reasoning":{"evidence":"blur","solution":"sharpen"},"rating":3.0}'
        )
        self.assertEqual(numeric_rating["rating"], 3.0)

    def test_editor_response_must_be_successful_diffusers_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            edited = Path(temporary_directory) / "edited.png"
            edited.write_bytes(b"edited")
            resolved = validate_editor_response(
                {
                    "status": "success",
                    "backend": "diffusers_flux2",
                    "edited_path": str(edited),
                }
            )
            self.assertEqual(resolved, edited.resolve())

        with self.assertRaises(EditorServiceError):
            validate_editor_response(
                {"status": "error", "backend": "diffusers_flux2", "error": "failed"}
            )
        with self.assertRaises(EditorServiceError):
            validate_editor_response(
                {
                    "status": "success",
                    "backend": "unexpected",
                    "edited_path": "/missing.png",
                }
            )

    def test_atomic_json_replaces_complete_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "result.json"
            atomic_write_json(destination, {"version": 1})
            atomic_write_json(destination, {"version": 2, "text": "完整"})
            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                '{\n  "version": 2,\n  "text": "完整"\n}\n',
            )
            self.assertEqual(list(destination.parent.glob(".*.tmp")), [])

    def test_run_example_writes_reproducible_artifacts_with_fakes(self) -> None:
        completion = (
            '{"reasoning":{"evidence":"Noise is visible in the wall shadows.",'
            '"solution":"Reduce shadow noise while preserving brick texture."},'
            '"rating":"3.10"}'
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "input.png"
            edited = root / "server-edited.png"
            output = root / "result"
            source.write_bytes(b"source image")
            edited.write_bytes(b"edited image")
            args = parse_args(
                [
                    "--image",
                    str(source),
                    "--output-dir",
                    str(output),
                    "--editor-url",
                    "http://127.0.0.1:19000/",
                    "--seed",
                    "99",
                ]
            )
            observed: dict[str, object] = {}

            def fake_actor(_image_path, _args):
                return completion

            def fake_editor(url, request, timeout_sec):
                observed.update({"url": url, "request": request, "timeout": timeout_sec})
                return {
                    "status": "success",
                    "backend": "diffusers_flux2",
                    "edited_path": str(edited),
                    "seed": 99,
                    "profile_name": "test",
                }

            provenance = run_example(
                args,
                actor_generator=fake_actor,
                editor_post=fake_editor,
            )

            self.assertEqual(observed["url"], "http://127.0.0.1:19000/edit")
            self.assertEqual(
                observed["request"]["positive_prompt"],
                "Reduce shadow noise while preserving brick texture.",
            )
            self.assertTrue(provenance["editor"]["solution_forwarded_verbatim"])
            self.assertEqual((output / "actor_raw.txt").read_text(), completion)
            self.assertEqual((output / "source_image.png").read_bytes(), b"source image")
            self.assertEqual((output / "edited_image.png").read_bytes(), b"edited image")
            for name in (
                "actor_output.json",
                "editor_request.json",
                "editor_response.json",
                "provenance.json",
            ):
                self.assertTrue((output / name).is_file(), name)

    def test_run_example_preserves_invalid_raw_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "input.png"
            output = root / "result"
            source.write_bytes(b"source image")
            args = parse_args(
                [
                    "--image",
                    str(source),
                    "--output-dir",
                    str(output),
                ]
            )

            def unexpected_editor(*_args, **_kwargs):
                self.fail("Editor must not be called for an invalid Actor completion")

            with self.assertRaises(ActorOutputError):
                run_example(
                    args,
                    actor_generator=lambda _image, _args: "invalid completion",
                    editor_post=unexpected_editor,
                )

            self.assertEqual(
                (output / "actor_raw.txt").read_text(encoding="utf-8"),
                "invalid completion",
            )
            self.assertFalse((output / "editor_request.json").exists())


if __name__ == "__main__":
    unittest.main()
