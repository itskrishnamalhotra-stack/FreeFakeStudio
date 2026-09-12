import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import startup_restore


class StartupRestoreTests(unittest.TestCase):
    def fingerprint(self, suffix="a"):
        return {
            "schema": 1,
            "python": "3.12.1",
            "python_abi": "cpython-312",
            "platform": "linux",
            "machine": "x86_64",
            "torch": "2.8",
            "cuda": "12.6",
            "cudnn": "9",
            "gpu": "Tesla T4",
            "gpu_capability": "7.5",
            "driver": f"550.{suffix}",
            "id": suffix,
        }

    def test_fingerprint_rejects_driver_or_runtime_changes(self):
        first = self.fingerprint("a")
        second = self.fingerprint("b")
        self.assertTrue(startup_restore.fingerprints_match(first, dict(first)))
        self.assertFalse(startup_restore.fingerprints_match(first, second))

    def test_source_bundle_round_trip_and_reuse(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source"
            source.mkdir()
            (source / "app.py").write_text("print('ready')", encoding="utf-8")
            (source / ".git").mkdir()
            (source / ".git" / "noise").write_text("skip", encoding="utf-8")
            destination = root / "runtime" / "app"
            bundles = root / "drive" / "bundles"
            manifests = root / "drive" / "manifests"

            state, manifest = startup_restore.ensure_source_bundle(
                "app",
                source,
                destination,
                bundles,
                manifests,
                "revision-a",
                exclude_names={".git"},
            )
            self.assertEqual(state, "rebuilt")
            self.assertEqual((destination / "app.py").read_text(encoding="utf-8"), "print('ready')")
            self.assertFalse((destination / ".git").exists())

            state, reused = startup_restore.ensure_source_bundle(
                "app",
                source,
                destination,
                bundles,
                manifests,
                "revision-a",
                exclude_names={".git"},
            )
            self.assertEqual(state, "restored")
            self.assertEqual(reused["archive_sha256"], manifest["archive_sha256"])

    def test_corrupt_source_archive_is_rebuilt_automatically(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source"
            source.mkdir()
            (source / "value.txt").write_text("good", encoding="utf-8")
            destination = root / "runtime" / "source"
            bundles = root / "bundles"
            manifests = root / "manifests"
            startup_restore.ensure_source_bundle(
                "source", source, destination, bundles, manifests, "same"
            )
            archive = bundles / "source.tar.gz"
            archive.write_bytes(b"broken")

            state, _ = startup_restore.ensure_source_bundle(
                "source", source, destination, bundles, manifests, "same"
            )
            self.assertEqual(state, "rebuilt")
            self.assertEqual((destination / "value.txt").read_text(encoding="utf-8"), "good")

    def test_source_signature_detects_local_edits_but_ignores_models(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "nodes.py").write_text("one", encoding="utf-8")
            models = root / "models"
            models.mkdir()
            (models / "large.bin").write_bytes(b"model")
            first = startup_restore.source_signature(root, exclude_prefixes={"models"})
            (models / "large.bin").write_bytes(b"changed model")
            self.assertEqual(
                startup_restore.source_signature(root, exclude_prefixes={"models"}),
                first,
            )
            (root / "nodes.py").write_text("a changed implementation", encoding="utf-8")
            self.assertNotEqual(
                startup_restore.source_signature(root, exclude_prefixes={"models"}),
                first,
            )

    def test_safe_extract_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            archive_path = root / "bad.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                payload = b"bad"
                member = tarfile.TarInfo("../escape.txt")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            with self.assertRaisesRegex(RuntimeError, "Unsafe archive member"):
                startup_restore.restore_archive(
                    archive_path,
                    root / "runtime" / "target",
                    allowed_root=root / "runtime",
                )
            self.assertFalse((root / "escape.txt").exists())

    def test_cache_bundle_is_invalidated_by_fingerprint(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            cache = root / "runtime" / "cache"
            cache.mkdir(parents=True)
            (cache / "kernel.bin").write_bytes(b"compiled")
            archive = root / "drive" / "cache.tar.gz"
            manifest = root / "drive" / "cache.json"
            startup_restore.snapshot_cache_bundle(cache, archive, manifest, self.fingerprint("a"))

            self.assertEqual(
                startup_restore.restore_cache_bundle(
                    cache, archive, manifest, self.fingerprint("a")
                ),
                "restored",
            )
            self.assertTrue((cache / "kernel.bin").is_file())
            self.assertEqual(
                startup_restore.restore_cache_bundle(
                    cache, archive, manifest, self.fingerprint("b")
                ),
                "fresh",
            )
            self.assertFalse((cache / "kernel.bin").exists())

    def test_corrupt_cache_archive_falls_back_to_fresh(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            cache = root / "runtime" / "cache"
            cache.mkdir(parents=True)
            (cache / "kernel.bin").write_bytes(b"compiled")
            archive = root / "drive" / "cache.tar.gz"
            manifest = root / "drive" / "cache.json"
            startup_restore.snapshot_cache_bundle(cache, archive, manifest, self.fingerprint())
            archive.write_bytes(b"corrupt")
            state = startup_restore.restore_cache_bundle(
                cache, archive, manifest, self.fingerprint()
            )
            self.assertEqual(state, "fresh")
            self.assertFalse((cache / "kernel.bin").exists())

    def test_unchanged_cache_reuses_existing_archive(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            cache = root / "runtime" / "cache"
            cache.mkdir(parents=True)
            (cache / "kernel.bin").write_bytes(b"compiled")
            archive = root / "drive" / "cache.tar.gz"
            manifest = root / "drive" / "cache.json"
            first = startup_restore.snapshot_cache_bundle(
                cache, archive, manifest, self.fingerprint("a")
            )
            first_mtime = archive.stat().st_mtime_ns
            second = startup_restore.snapshot_cache_bundle(
                cache, archive, manifest, self.fingerprint("a")
            )

            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertEqual(archive.stat().st_mtime_ns, first_mtime)
            third = startup_restore.snapshot_cache_bundle(
                cache, archive, manifest, self.fingerprint("a"), force=True
            )
            self.assertFalse(third["reused"])

    def test_cache_environment_sets_all_supported_paths(self):
        with tempfile.TemporaryDirectory() as root:
            values = startup_restore.cache_environment(Path(root) / "cache")
            for key in (
                "HF_HOME",
                "HF_HUB_CACHE",
                "HF_XET_CACHE",
                "HF_ASSETS_CACHE",
                "TORCH_HOME",
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "TORCH_EXTENSIONS_DIR",
                "CUDA_CACHE_PATH",
            ):
                self.assertIn(key, values)
                self.assertTrue(Path(values[key]).is_dir())

    def test_model_staging_copies_selected_and_links_others(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            drive = root / "drive_models"
            local = root / "runtime" / "models"
            selected = drive / "diffusion_models" / "flux.bin"
            other = drive / "diffusion_models" / "other.bin"
            selected.parent.mkdir(parents=True)
            selected.write_bytes(b"flux")
            other.write_bytes(b"other")
            fake_usage = os.statvfs if False else mock.Mock(free=100 * 1024**3)
            links = []
            def fake_symlink(source, destination):
                links.append((source, destination))
                Path(destination).write_text("linked", encoding="utf-8")

            with mock.patch("startup_restore.shutil.disk_usage", return_value=fake_usage), \
                    mock.patch("startup_restore.os.symlink", side_effect=fake_symlink):
                report = startup_restore.mirror_model_tree(
                    drive, local, [selected], copy_to_ssd=True
                )
            modes = {Path(item["source"]).name: item["mode"] for item in report}
            self.assertEqual(modes, {"flux.bin": "copied", "other.bin": "linked"})
            self.assertFalse((local / "diffusion_models" / "flux.bin").is_symlink())
            self.assertEqual(len(links), 1)
            self.assertTrue(links[0][1].endswith("other.bin"))

    def test_model_staging_falls_back_to_link_when_disk_is_low(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "drive" / "model.bin"
            source.parent.mkdir()
            source.write_bytes(b"model")
            destination = root / "runtime" / "model.bin"
            with mock.patch(
                "startup_restore.shutil.disk_usage", return_value=mock.Mock(free=0)
            ), mock.patch("startup_restore.os.symlink") as symlink:
                mode = startup_restore.stage_file(source, destination, True)
            self.assertEqual(mode, "linked")
            symlink.assert_called_once_with(str(source.resolve()), str(destination))

    def test_required_model_is_mirrored_when_drive_listing_omits_it(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            drive = (root / "drive_models").resolve()
            local = root / "runtime" / "models"
            required = drive / "text_encoders" / "required.bin"
            required.parent.mkdir(parents=True)
            required.write_bytes(b"required")

            links = []

            def fake_symlink(source, destination):
                links.append((source, destination))
                Path(destination).write_bytes(Path(source).read_bytes())

            with mock.patch("startup_restore.os.walk", return_value=[]), \
                    mock.patch("startup_restore.os.symlink", side_effect=fake_symlink):
                report = startup_restore.mirror_model_tree(
                    drive,
                    local,
                    copy_to_ssd=False,
                    required_sources=[required],
                )

            destination = local / "text_encoders" / "required.bin"
            self.assertTrue(destination.is_file())
            self.assertEqual(len(links), 1)
            self.assertEqual(report[0]["source"], str(required))

    def test_readiness_file_is_atomic_and_parseable(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "ready.json"
            startup_restore.write_readiness(path, ready=True, warmup="complete")
            payload = startup_restore.readiness_payload(path)
            self.assertTrue(payload["ready"])
            self.assertEqual(payload["warmup"], "complete")
            self.assertFalse(path.with_name("ready.json.part").exists())

    def test_corrupt_wheel_is_removed(self):
        with tempfile.TemporaryDirectory() as root:
            wheel = Path(root) / "bad-1-py3-none-any.whl"
            wheel.write_bytes(b"not zip")
            self.assertEqual(startup_restore.validate_wheelhouse(root), [wheel.name])
            self.assertFalse(wheel.exists())

    def test_environment_overlay_dependency_closure_excludes_torch_stack(self):
        distributions = {
            "transformers": mock.Mock(
                metadata={"Name": "transformers"},
                requires=["tokenizers>=1", "torch>=2", "pytest; extra == 'test'"],
            ),
            "tokenizers": mock.Mock(metadata={"Name": "tokenizers"}, requires=[]),
        }

        def lookup(name):
            key = str(name).lower()
            if key not in distributions:
                raise startup_restore.importlib.metadata.PackageNotFoundError(name)
            return distributions[key]

        with mock.patch("startup_restore.importlib.metadata.distribution", side_effect=lookup):
            closure = startup_restore.distribution_closure(["transformers"])

        self.assertEqual(closure, ["transformers", "tokenizers"])

    def test_environment_overlay_restores_only_for_matching_requirements(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source_overlay"
            source.mkdir()
            (source / "package.py").write_text("READY = True", encoding="utf-8")
            archive = root / "drive" / "environment.tar.gz"
            details = startup_restore.create_archive(source, archive)
            manifest_path = root / "drive" / "environment.json"
            startup_restore.atomic_write_json(
                manifest_path,
                {
                    "schema": 1,
                    "fingerprint": self.fingerprint("a"),
                    "requirements_digest": "requirements-a",
                    "archive_sha256": details["sha256"],
                },
            )
            destination = root / "runtime" / "python_packages"
            self.assertTrue(
                startup_restore.restore_environment_bundle(
                    destination,
                    archive,
                    manifest_path,
                    self.fingerprint("a"),
                    "requirements-a",
                )
            )
            self.assertTrue((destination / "package.py").is_file())
            self.assertFalse(
                startup_restore.restore_environment_bundle(
                    destination,
                    archive,
                    manifest_path,
                    self.fingerprint("a"),
                    "requirements-b",
                )
            )
            self.assertFalse(destination.exists())

    def test_corrupt_environment_archive_triggers_rebuild_path(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source"
            source.mkdir()
            (source / "package.py").write_text("READY = True", encoding="utf-8")
            archive = root / "drive" / "environment.tar.gz"
            details = startup_restore.create_archive(source, archive)
            manifest = root / "drive" / "environment.json"
            startup_restore.atomic_write_json(
                manifest,
                {
                    "schema": 1,
                    "fingerprint": self.fingerprint(),
                    "requirements_digest": "req",
                    "archive_sha256": details["sha256"],
                },
            )
            archive.write_bytes(b"corrupt")
            destination = root / "runtime" / "python_packages"
            self.assertFalse(
                startup_restore.restore_environment_bundle(
                    destination, archive, manifest, self.fingerprint(), "req"
                )
            )


if __name__ == "__main__":
    unittest.main()
