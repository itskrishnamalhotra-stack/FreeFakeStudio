"""Vision dependency recovery tests: no real torch imports, weights or downloads."""
import ast
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import avatar_vision
import vision_runtime
import web_server


def launcher_function(name, namespace):
    tree = ast.parse(Path("launch.py").read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "launch.py", "exec"), namespace)
    return namespace[name]


class VisionBackendTests(unittest.TestCase):
    def setUp(self):
        self.torch = mock.MagicMock()
        self.torch.cuda.is_available.return_value = True
        self.torch.inference_mode.side_effect = nullcontext
        self.sample = self.torch.linspace.return_value.reshape.return_value
        self.sample.shape = (16, 16)
        self.functional = mock.Mock()
        self.functional.quantize_4bit.return_value = ("packed", "state")
        self.functional.dequantize_4bit.return_value.shape = (16, 16)
        self.quantizer = mock.Mock()
        self.transformers = SimpleNamespace(AutoModelForImageTextToText=mock.Mock(),
                                            AutoProcessor=mock.Mock(), BitsAndBytesConfig=mock.Mock())
        modules = {"torch": self.torch, "transformers": self.transformers,
                   "transformers.quantizers.quantizer_bnb_4bit": SimpleNamespace(Bnb4BitHfQuantizer=self.quantizer),
                   "bitsandbytes.functional": self.functional}
        self.enterContext(mock.patch.dict(sys.modules, modules))
        versions = {"transformers": "5.13.0", "accelerate": "1.11.0", "bitsandbytes": "0.46.1", "sentencepiece": "0.2.1"}
        self.enterContext(mock.patch.object(vision_runtime.metadata, "version", side_effect=lambda name: versions[name]))
        self.enterContext(mock.patch.object(vision_runtime, "diagnostics", return_value={"packages": {}}))

    def test_exercises_real_quantizer_contract_and_cuda_nf4_without_weights(self):
        self.assertTrue(vision_runtime.validate()["ok"])
        self.quantizer.return_value.validate_environment.assert_called_once_with(device_map={"": 0})
        self.functional.quantize_4bit.assert_called_once_with(self.sample, quant_type="nf4")
        self.torch.cuda.synchronize.assert_called_once()
        self.transformers.AutoProcessor.from_pretrained.assert_not_called()
        self.transformers.AutoModelForImageTextToText.from_pretrained.assert_not_called()
        config = self.transformers.BitsAndBytesConfig.call_args.kwargs
        self.assertTrue(config["load_in_4bit"])
        self.assertEqual(config["bnb_4bit_quant_type"], "nf4")
        self.assertIs(config["bnb_4bit_compute_dtype"], self.torch.float16)

    def test_same_error_as_reported_is_not_mistaken_for_a_working_install(self):
        self.quantizer.return_value.validate_environment.side_effect = ImportError("4-bit quantization requires bitsandbytes")
        with self.assertRaisesRegex(ImportError, "requires bitsandbytes"):
            vision_runtime.validate()
        self.functional.quantize_4bit.assert_not_called()

    def test_missing_cuda_fails_before_backend_or_downloads(self):
        self.torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, "CUDA GPU"):
            vision_runtime.validate()
        self.quantizer.assert_not_called()

    def test_old_package_is_not_accepted_just_because_it_imports(self):
        vision_runtime.metadata.version.side_effect = lambda name: "0.45.0" if name == "bitsandbytes" else {"transformers": "5.13.0", "accelerate": "1.11.0", "sentencepiece": "0.2.1"}[name]
        with self.assertRaisesRegex(RuntimeError, "bitsandbytes 0.45.0"):
            vision_runtime.validate()

    def test_invalid_cuda_result_is_rejected(self):
        self.torch.isfinite.return_value.all.return_value.item.return_value = False
        with self.assertRaisesRegex(RuntimeError, "invalid values"):
            vision_runtime.validate()

    def test_missing_native_kernel_is_not_silently_ignored(self):
        self.functional.quantize_4bit.side_effect = RuntimeError("CUDA kernel unavailable")
        with self.assertRaisesRegex(RuntimeError, "kernel unavailable"):
            vision_runtime.validate()

    def test_loader_checks_backend_before_processor_download(self):
        with mock.patch.object(vision_runtime, "validate", side_effect=ImportError("requires bitsandbytes")), \
                mock.patch.object(avatar_vision, "_write_vision_diagnostic", return_value="vision-error.json") as diagnostic:
            with self.assertRaisesRegex(RuntimeError, "saved batch is preserved"):
                avatar_vision._load_model.__wrapped__()
        self.transformers.AutoProcessor.from_pretrained.assert_not_called()
        self.assertEqual(diagnostic.call_args.args[0]["context"], "model-load")
        self.assertIn("requires bitsandbytes", diagnostic.call_args.args[0]["traceback"])

    def test_loader_keeps_nf4_instead_of_falling_back_to_larger_model(self):
        with mock.patch.object(vision_runtime, "validate") as validate, mock.patch.object(avatar_vision, "cleanup_memory"):
            avatar_vision._load_model.__wrapped__()
        validate.assert_called_once()
        kwargs = self.transformers.AutoModelForImageTextToText.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["device_map"], {"": 0})
        self.assertIs(kwargs["dtype"], self.torch.float16)
        self.assertIs(kwargs["quantization_config"], self.transformers.BitsAndBytesConfig.return_value)


class LauncherVisionTests(unittest.TestCase):
    def test_probe_includes_both_output_streams_and_has_timeout(self):
        process = mock.Mock()
        process.run.return_value = SimpleNamespace(returncode=1, stdout="info", stderr="actual error")
        process.TimeoutExpired = subprocess.TimeoutExpired
        probe = launcher_function("avatar_quantization_probe", {"sys": sys, "subprocess": process, "DRIVE_APP": Path("app")})
        self.assertEqual(probe(), (False, "info\nactual error"))
        args = process.run.call_args
        self.assertEqual(args.kwargs["timeout"], 120)
        self.assertIn("from vision_runtime import main", args.args[0][-1])

    def test_probe_timeout_returns_actionable_failure(self):
        process = mock.Mock()
        process.TimeoutExpired = subprocess.TimeoutExpired
        process.run.side_effect = subprocess.TimeoutExpired("probe", 120)
        probe = launcher_function("avatar_quantization_probe", {"sys": sys, "subprocess": process, "DRIVE_APP": Path("app")})
        self.assertEqual(probe(), (False, "Vision CUDA NF4 validation timed out after 120 seconds"))

    def repair(self, root, responses):
        namespace = {"avatar_quantization_probe": mock.Mock(side_effect=responses),
                     "discard_restored_environment": mock.Mock(), "pip_install": mock.Mock(),
                     "WS": Path(root), "MANIFESTS": Path(root)}
        return launcher_function("ensure_avatar_quantization", namespace), namespace

    def test_healthy_restore_is_not_reinstalled(self):
        with tempfile.TemporaryDirectory() as root:
            repair, ns = self.repair(root, [(True, "OK")])
            self.assertTrue(repair(True))
            ns["pip_install"].assert_not_called()
            ns["discard_restored_environment"].assert_not_called()

    def test_invalid_overlay_is_removed_before_touching_global_packages(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = Path(root) / "python-overlay.json"
            manifest.write_text("{}", encoding="utf-8")
            repair, ns = self.repair(root, [(False, "stale overlay"), (True, "global OK")])
            self.assertFalse(repair(True))
            ns["discard_restored_environment"].assert_called_once()
            ns["pip_install"].assert_not_called()
            self.assertFalse(manifest.exists())

    def test_repairs_broken_metadata_without_force_reinstalling_torch_dependencies(self):
        with tempfile.TemporaryDirectory() as root:
            repair, ns = self.repair(root, [(False, "broken"), (False, "broken"), (True, "fixed")])
            self.assertFalse(repair(False))
            self.assertEqual(ns["pip_install"].call_count, 2)
            first, second = ns["pip_install"].call_args_list
            self.assertEqual(first.args[:2], ("--force-reinstall", "--no-deps"))
            self.assertEqual(second.args, vision_runtime.REQUIREMENTS)
            self.assertNotIn("torch", first.args)
            self.assertEqual((Path(root) / "results/_debug/vision-dependencies.txt").read_text(), "fixed")

    def test_failed_repair_blocks_link_with_diagnostic_and_no_deferred_success(self):
        with tempfile.TemporaryDirectory() as root:
            repair, ns = self.repair(root, [(False, "broken")] * 3)
            with self.assertRaisesRegex(RuntimeError, "No app link was published"):
                repair(True)
            self.assertEqual(ns["avatar_quantization_probe"].call_count, 3)
            self.assertEqual((Path(root) / "results/_debug/vision-dependencies.txt").read_text(), "broken")

    def test_checks_run_after_comfy_changes_and_final_overlay_build(self):
        source = Path("launch.py").read_text(encoding="utf-8")
        final = source.rindex("environment_restored = ensure_avatar_quantization(environment_restored)")
        self.assertGreater(final, source.rindex("environment_manifest = snapshot_prepared_environment"))
        self.assertGreater(final, source.rindex("runtime_probe = verify_comfy_runtime()"))
        self.assertLess(final, source.rindex("launch_app_process("))


class VisionReadinessTests(unittest.TestCase):
    def setUp(self):
        self.folder = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.dict(os.environ, {"FFS_READY_STATE_PATH": str(Path(self.folder) / "ready.json"),
            "FFS_PRELOAD_FLUX": "1", "FFS_PRELOAD_AVATAR_VISION": "1", "FFS_WARMUP_ENABLED": "0", "FFS_EXPOSE_AFTER_WARMUP": "1"}))
        self.enterContext(mock.patch.object(web_server.model_manager, "DEV_MODE", False))
        self.enterContext(mock.patch.object(web_server.model_manager, "check_model_files"))
        self.engine = self.enterContext(mock.patch.object(web_server.model_manager, "ensure_model"))
        self.enterContext(mock.patch.object(web_server.workspace, "get_save_dir", return_value=self.folder))
        self.preload = self.enterContext(mock.patch.object(avatar_vision, "preload"))
        self.warmup = self.enterContext(mock.patch.object(avatar_vision, "warmup"))

    def ready(self):
        import json
        return json.loads((Path(self.folder) / "ready.json").read_text())

    def test_requested_preload_error_is_not_published_as_ready(self):
        self.preload.side_effect = ImportError("requires bitsandbytes")
        with self.assertRaisesRegex(RuntimeError, "No app link was published"):
            web_server.prepare_models()
        self.assertFalse(self.ready()["ready"])
        self.assertEqual(self.ready()["vision"], "failed")
        self.assertIn("requires bitsandbytes", (Path(self.folder) / "_debug/vision-startup.txt").read_text())

    def test_success_is_marked_loaded_only_after_vision_load(self):
        web_server.prepare_models()
        self.preload.assert_called_once()
        self.assertTrue(self.ready()["ready"])
        self.assertEqual(self.ready()["vision"], "loaded")

    def test_vision_preload_is_respected_even_when_flux_preload_is_disabled(self):
        with mock.patch.dict(os.environ, {"FFS_PRELOAD_FLUX": "0", "FFS_EXPOSE_AFTER_WARMUP": "0"}):
            web_server.prepare_models()
        self.preload.assert_called_once()
        self.engine.assert_not_called()

    def test_explicitly_disabled_vision_is_not_loaded(self):
        with mock.patch.dict(os.environ, {"FFS_PRELOAD_AVATAR_VISION": "0"}):
            web_server.prepare_models()
        self.preload.assert_not_called()
        self.assertEqual(self.ready()["vision"], "disabled")

    def test_failed_warmup_also_blocks_readiness(self):
        self.warmup.side_effect = RuntimeError("empty answer")
        with mock.patch.dict(os.environ, {"FFS_WARMUP_ENABLED": "1"}):
            with self.assertRaisesRegex(RuntimeError, "empty answer"):
                web_server.prepare_models()
        self.assertFalse(self.ready()["ready"])

    def test_development_preview_never_imports_or_loads_real_vision(self):
        with mock.patch.object(web_server.model_manager, "DEV_MODE", True), \
                mock.patch.object(web_server.model_manager, "set_model_availability"):
            web_server.prepare_models()
        self.preload.assert_not_called()
        self.warmup.assert_not_called()
        self.engine.assert_not_called()
        self.assertEqual(self.ready()["phase"], "development")


if __name__ == "__main__":
    unittest.main()
