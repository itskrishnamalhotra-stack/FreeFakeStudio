import ast
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import engine_z_image
import model_manager


class _Loader:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def load_unet(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return (self.result,)

    def load_clip(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return (self.result,)

    def load_vae(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return (self.result,)


class _Aura:
    def __init__(self):
        self.calls = []

    def patch_aura(self, *args):
        self.calls.append(args)
        return ("patched-unet",)


class ZImageMemoryTests(unittest.TestCase):
    def tearDown(self):
        engine_z_image._loaded = False
        engine_z_image._unet = None
        engine_z_image._clip = None
        engine_z_image._vae = None

    def test_load_uses_low_ram_encoder(self):
        unet = _Loader("raw-unet")
        clip = _Loader("clip")
        vae = _Loader("vae")
        aura = _Aura()
        nodes = {
            "UNETLoader": unet,
            "CLIPLoader": clip,
            "VAELoader": vae,
            "ModelSamplingAuraFlow": aura,
        }

        with mock.patch.object(engine_z_image, "_get_nodes", return_value=nodes), \
             mock.patch.object(engine_z_image, "_require_host_headroom"), \
             mock.patch.object(engine_z_image, "_memory_status"), \
             mock.patch("builtins.print"):
            engine_z_image.load()

        self.assertEqual(
            unet.calls[0][0],
            ("z-image-turbo-fp8-e4m3fn.safetensors", "fp8_e4m3fn_fast"),
        )
        self.assertEqual(
            clip.calls[0],
            (("qwen_3_4b_fp4_mixed.safetensors",), {"type": "lumina2"}),
        )
        self.assertEqual(vae.calls[0][0], ("ae.safetensors",))
        self.assertEqual(aura.calls[0], ("raw-unet", 3.0))
        self.assertTrue(engine_z_image.is_loaded())

    def test_comfy_memory_defaults(self):
        args = types.SimpleNamespace()
        cli_args = types.ModuleType("comfy.cli_args")
        cli_args.args = args
        comfy = types.ModuleType("comfy")
        comfy.cli_args = cli_args

        with mock.patch.dict(
            sys.modules,
            {"comfy": comfy, "comfy.cli_args": cli_args},
        ):
            engine_z_image._configure_comfy_memory()

        self.assertTrue(args.cache_none)
        self.assertTrue(args.enable_dynamic_vram)
        self.assertTrue(args.disable_pinned_memory)
        self.assertFalse(args.high_ram)
        self.assertFalse(args.disable_dynamic_vram)

    def test_manager_releases_comfy_registry(self):
        calls = []
        model_management = types.ModuleType("comfy.model_management")
        model_management.unload_all_models = lambda: calls.append("unload")
        model_management.soft_empty_cache = lambda: calls.append("empty")
        comfy = types.ModuleType("comfy")
        comfy.model_management = model_management

        cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: calls.append("cuda-empty"),
        )
        torch = types.ModuleType("torch")
        torch.cuda = cuda

        with mock.patch.dict(
            sys.modules,
            {
                "comfy": comfy,
                "comfy.model_management": model_management,
                "torch": torch,
            },
        ):
            model_manager._clear_memory()

        self.assertEqual(calls, ["unload", "empty"])


class LauncherRepositoryTests(unittest.TestCase):
    @staticmethod
    def _ensure_repo(fake_run_cmd):
        source = Path("launch.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "ensure_repo"
        )
        namespace = {"run_cmd": fake_run_cmd}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "launch.py", "exec"), namespace)
        return namespace["ensure_repo"]

    def test_empty_comfy_scaffold_is_replaced_by_clone(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "ComfyUI"
            (target / "models" / "diffusion_models").mkdir(parents=True)
            (target / "models" / "vae").mkdir(parents=True)

            def fake_run_cmd(args, quiet=True):
                self.assertEqual(args[1], "clone")
                self.assertFalse(target.exists())
                (target / ".git").mkdir(parents=True)

            ensure_repo = self._ensure_repo(fake_run_cmd)
            result = ensure_repo(target, "https://example.test/ComfyUI.git", "v0.28.0")

            self.assertEqual(result, "installed")
            self.assertTrue((target / ".git").is_dir())

    def test_non_git_directory_with_files_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "ComfyUI"
            target.mkdir()
            marker = target / "keep-me.txt"
            marker.write_text("persistent", encoding="utf-8")
            ensure_repo = self._ensure_repo(lambda *args, **kwargs: None)

            with self.assertRaisesRegex(RuntimeError, "left it untouched"):
                ensure_repo(target, "https://example.test/ComfyUI.git", "v0.28.0")

            self.assertEqual(marker.read_text(encoding="utf-8"), "persistent")


if __name__ == "__main__":
    unittest.main()
