import ast
import base64
import html
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from urllib.parse import unquote, urlparse
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

import engine_flux_klein_4b
import engine_z_image
import gguf_nodes
import model_manager


def _source_function(source_path, name, namespace):
    source = Path(source_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), source_path, "exec"), namespace)
    return namespace[name]


class StudioUiRegressionTests(unittest.TestCase):
    def test_smart_image_edit_does_not_force_a_face_mask(self):
        def unexpected_mask(_image):
            raise AssertionError("ordinary img2img must not invoke automatic masks")

        select_mask = _source_function(
            "app.py",
            "_select_mask_for_prompt",
            {
                "re": __import__("re"),
                "np": np,
                "auto_mask_background": unexpected_mask,
                "auto_mask_except_face": unexpected_mask,
            },
        )
        image = Image.new("RGB", (64, 64), "white")

        mask, prompt, denoise = select_mask("turn this into a watercolor", image)

        self.assertIsNone(mask)
        self.assertEqual(prompt, "turn this into a watercolor")
        self.assertEqual(denoise, 1.0)

    def test_partial_cv2_install_uses_safe_face_fallback(self):
        partial_cv2 = types.ModuleType("cv2")
        namespace = {
            "DEV_MODE": False,
            "np": np,
            "os": os,
            "Image": Image,
            "ImageDraw": ImageDraw,
            "auto_mask_background": lambda image: np.zeros(
                (image.height, image.width), dtype=np.uint8
            ),
        }
        auto_mask = _source_function("app.py", "auto_mask_except_face", namespace)

        with mock.patch.dict(sys.modules, {"cv2": partial_cv2}):
            mask = auto_mask(Image.new("RGB", (96, 128), "white"))

        self.assertEqual(mask.shape, (128, 96))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertGreater(np.count_nonzero(mask == 0), 0)

    def test_chat_turn_contains_reference_and_generated_images(self):
        namespace = {
            "Image": Image,
            "BytesIO": __import__("io").BytesIO,
            "base64": base64,
            "html": html,
            "os": os,
            "quote": __import__("urllib.parse", fromlist=["quote"]).quote,
            "_ui_trace": lambda _message: None,
        }
        thumbnail = _source_function("app.py", "_image_thumbnail_data_uri", namespace)
        namespace["_image_thumbnail_data_uri"] = thumbnail
        namespace["_as_image_list"] = _source_function("app.py", "_as_image_list", namespace)
        request_html = _source_function("app.py", "_request_html", namespace)
        assistant_html = _source_function("app.py", "_assistant_html", namespace)

        request = request_html(
            "change the background", "FLUX.2-klein 4B",
            [
                Image.new("RGB", (32, 32), "red"),
                Image.new("RGB", (32, 32), "blue"),
            ],
            "Background Only",
        )
        response = assistant_html(["results/example image.png"], "123")

        self.assertIn("data:image/jpeg;base64,", request)
        self.assertIn(">Reference<", request)
        self.assertIn("Image 1", request)
        self.assertIn("Image 2", request)
        self.assertEqual(request.count("data:image/jpeg;base64,"), 2)
        self.assertIn("Background Only", request)
        self.assertIn('<img src="/gradio_api/file=', response)
        self.assertIn("Seed 123", response)

    def test_flux_reference_conditioning_appends_every_image(self):
        class FakeVaeEncode:
            def encode(self, _vae, image):
                return ({"samples": image},)

        class FakeReferenceLatent:
            def append(self, conditioning, latent):
                return (conditioning + [latent],)

        nodes = {
            "VAEEncode": FakeVaeEncode(),
            "ReferenceLatent": FakeReferenceLatent(),
        }
        references = [
            Image.new("RGB", (640, 480), "red"),
            Image.new("RGB", (480, 640), "blue"),
            Image.new("RGB", (512, 512), "green"),
        ]

        with mock.patch.object(engine_flux_klein_4b, "_pil_to_tensor", side_effect=lambda image: image), \
             mock.patch.object(engine_flux_klein_4b, "_vae", object()), \
             mock.patch("builtins.print"):
            conditioned = engine_flux_klein_4b._add_reference_conditioning(
                nodes, [], references
            )

        self.assertEqual(len(conditioned), 3)
        for latent in conditioned:
            self.assertLessEqual(latent["samples"].width * latent["samples"].height, 1024**2 // 3)

    def test_flux_reference_limit_is_four(self):
        images = [Image.new("RGB", (32, 32)) for _ in range(5)]
        with self.assertRaisesRegex(ValueError, "at most 4"):
            engine_flux_klein_4b._normalize_references(images)


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
            "UnetLoaderGGUF": unet,
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
            ("z_image_turbo-Q3_K_M.gguf",),
        )
        self.assertEqual(
            clip.calls[0],
            (("qwen_3_4b_fp4_mixed.safetensors",), {"type": "lumina2"}),
        )
        self.assertEqual(vae.calls[0][0], ("ae.safetensors",))
        self.assertEqual(aura.calls[0], ("raw-unet", 3.0))
        self.assertTrue(engine_z_image.is_loaded())

    def test_registry_requires_z_image_gguf(self):
        info = model_manager.MODEL_REGISTRY["Z-Image Turbo"]
        self.assertEqual(info["model_file"], "z_image_turbo-Q3_K_M.gguf")
        self.assertIn(
            ("diffusion_models", "z_image_turbo-Q3_K_M.gguf"),
            info["required_files"],
        )

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

    def test_gguf_nodes_are_loaded_as_a_package(self):
        with tempfile.TemporaryDirectory() as root:
            package_dir = Path(root) / "custom_nodes" / "ComfyUI-GGUF"
            package_dir.mkdir(parents=True)
            (package_dir / "__init__.py").write_text(
                "from .nodes import NODE_CLASS_MAPPINGS\n",
                encoding="utf-8",
            )
            (package_dir / "nodes.py").write_text(
                "from .ops import Loader\nNODE_CLASS_MAPPINGS = {"
                "'UnetLoaderGGUF': Loader, 'CLIPLoaderGGUF': Loader}\n",
                encoding="utf-8",
            )
            (package_dir / "ops.py").write_text("class Loader: pass\n", encoding="utf-8")

            for name in tuple(sys.modules):
                if name == gguf_nodes.PACKAGE_NAME or name.startswith(gguf_nodes.PACKAGE_NAME + "."):
                    sys.modules.pop(name, None)
            try:
                mappings = gguf_nodes.load_gguf_node_mappings(root)
                self.assertIn("UnetLoaderGGUF", mappings)
                self.assertIn("CLIPLoaderGGUF", mappings)
                self.assertEqual(mappings["UnetLoaderGGUF"].__module__, gguf_nodes.PACKAGE_NAME + ".ops")
            finally:
                for name in tuple(sys.modules):
                    if name == gguf_nodes.PACKAGE_NAME or name.startswith(gguf_nodes.PACKAGE_NAME + "."):
                        sys.modules.pop(name, None)

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
    def _launch_function(name, namespace):
        source = Path("launch.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), "launch.py", "exec"), namespace)
        return namespace[name]

    @classmethod
    def _ensure_repo(cls, fake_run_cmd):
        return cls._launch_function("ensure_repo", {"run_cmd": fake_run_cmd})

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

    def test_custom_encoder_url_parser_accepts_huggingface_file(self):
        parse = self._launch_function(
            "parse_huggingface_file_url",
            {"urlparse": urlparse, "unquote": unquote, "Path": Path},
        )
        self.assertEqual(
            parse("https://huggingface.co/example/encoder/blob/main/model-q4_k_m.gguf"),
            ("example/encoder", "main", "model-q4_k_m.gguf", ".gguf"),
        )

    def test_custom_encoder_url_parser_rejects_arbitrary_host(self):
        parse = self._launch_function(
            "parse_huggingface_file_url",
            {"urlparse": urlparse, "unquote": unquote, "Path": Path},
        )
        with self.assertRaisesRegex(RuntimeError, "huggingface.co"):
            parse("https://example.test/model.gguf")

    def test_custom_encoder_metadata_rejects_oversized_file(self):
        sibling = types.SimpleNamespace(rfilename="model.gguf", size=4 * 1024**3)
        api = types.SimpleNamespace(
            model_info=lambda **kwargs: types.SimpleNamespace(siblings=[sibling])
        )
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = lambda: api
        function = self._launch_function(
            "huggingface_file_size",
            {
                "os": __import__("os"),
                "FLUX_CUSTOM_MIN_BYTES": 500 * 1024**2,
                "FLUX_CUSTOM_MAX_BYTES": 3 * 1024**3,
            },
        )
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}), \
             self.assertRaisesRegex(RuntimeError, "3.00 GiB"):
            function("example/encoder", "main", "model.gguf")

    def test_cached_model_is_moved_to_drive_without_symlink(self):
        import os
        import shutil

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            cached = root / "cache" / "snapshot" / "model.safetensors"
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b"complete-model")
            destination = root / "ComfyUI" / "models" / "diffusion_models"
            hub = types.ModuleType("huggingface_hub")
            hub.hf_hub_download = lambda **kwargs: str(cached)
            namespace = {
                "Path": Path,
                "CACHE": root / "cache",
                "REPAIR": False,
                "file_ok": lambda path: Path(path).is_file() and Path(path).stat().st_size > 0,
                "os": os,
                "shutil": shutil,
            }
            hub_download = self._launch_function("hub_download", namespace)

            with mock.patch.dict(sys.modules, {"huggingface_hub": hub}), \
                 mock.patch("os.symlink", side_effect=AssertionError("symlink must not be used")):
                target = hub_download("example/model", "model.safetensors", destination)

            self.assertEqual(target.read_bytes(), b"complete-model")
            self.assertFalse(cached.exists())

    def test_relative_cache_symlink_materializes_real_model_file(self):
        import os
        import shutil

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            blob = root / "cache" / "blobs" / "model-blob"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"real-model-data")
            cached = root / "cache" / "snapshots" / "revision" / "model.safetensors"
            cached.parent.mkdir(parents=True)
            cached.write_text("relative-link-placeholder", encoding="utf-8")

            destination = root / "ComfyUI" / "models" / "diffusion_models"
            hub = types.ModuleType("huggingface_hub")
            hub.hf_hub_download = lambda **kwargs: str(cached)
            namespace = {
                "Path": Path,
                "CACHE": root / "cache",
                "REPAIR": False,
                "file_ok": lambda path: Path(path).is_file() and Path(path).stat().st_size > 0,
                "os": os,
                "shutil": shutil,
            }
            hub_download = self._launch_function("hub_download", namespace)

            original_is_symlink = Path.is_symlink
            original_resolve = Path.resolve

            def fake_is_symlink(path):
                return True if path == cached else original_is_symlink(path)

            def fake_resolve(path, *args, **kwargs):
                return blob if path == cached else original_resolve(path, *args, **kwargs)

            with mock.patch.dict(sys.modules, {"huggingface_hub": hub}), \
                 mock.patch.object(Path, "is_symlink", fake_is_symlink), \
                 mock.patch.object(Path, "resolve", fake_resolve):
                target = hub_download("example/model", "model.safetensors", destination)

            self.assertTrue(target.is_file())
            self.assertFalse(target.is_symlink())
            self.assertEqual(target.read_bytes(), b"real-model-data")
            self.assertFalse(cached.is_symlink())


class FluxEncoderTests(unittest.TestCase):
    def tearDown(self):
        engine_flux_klein_4b._loaded = False
        engine_flux_klein_4b._unet = None
        engine_flux_klein_4b._clip = None
        engine_flux_klein_4b._vae = None
        engine_flux_klein_4b._loaded_encoder = None

    def test_official_flux_encoder_remains_default(self):
        clip = _Loader("official-clip")
        nodes = {
            "UNETLoader": _Loader("unet"),
            "CLIPLoader": clip,
            "VAELoader": _Loader("vae"),
        }
        with mock.patch.dict(__import__("os").environ, {"FFS_FLUX_ENCODER_MODE": "official"}), \
             mock.patch.object(engine_flux_klein_4b, "_get_nodes", return_value=nodes), \
             mock.patch("builtins.print"):
            engine_flux_klein_4b.load()

        self.assertEqual(
            clip.calls[0],
            (("qwen_3_4b_fp4_flux2.safetensors",), {"type": "flux2"}),
        )
        self.assertEqual(engine_flux_klein_4b.get_loaded_encoder(), "official")

    def test_distilled_flux_defaults_to_four_steps(self):
        self.assertEqual(
            model_manager.MODEL_REGISTRY["FLUX.2-klein 4B"]["default_steps"],
            4,
        )
        self.assertEqual(
            engine_flux_klein_4b.generate.__wrapped__.__defaults__[-1],
            4,
        )

    def test_custom_gguf_uses_gguf_clip_loader(self):
        import os

        with tempfile.TemporaryDirectory() as root:
            encoder_dir = Path(root) / "models" / "text_encoders"
            encoder_dir.mkdir(parents=True)
            (encoder_dir / "custom.gguf").write_bytes(b"GGUF")
            custom_clip = _Loader("custom-clip")
            nodes = {
                "UNETLoader": _Loader("unet"),
                "CLIPLoader": _Loader("official-clip"),
                "VAELoader": _Loader("vae"),
            }
            env = {
                "COMFYUI_ROOT": root,
                "FFS_FLUX_ENCODER_MODE": "custom",
                "FFS_FLUX_CUSTOM_ENCODER_FILE": "custom.gguf",
            }
            with mock.patch.dict(os.environ, env), \
                 mock.patch.object(engine_flux_klein_4b, "_get_nodes", return_value=nodes), \
                 mock.patch("gguf_nodes.load_gguf_node_mappings", return_value={
                     "CLIPLoaderGGUF": lambda: custom_clip,
                 }), \
                 mock.patch("builtins.print"):
                engine_flux_klein_4b.load()

            self.assertEqual(custom_clip.calls[0], (("custom.gguf",), {"type": "flux2"}))
            self.assertEqual(engine_flux_klein_4b.get_loaded_encoder(), "custom")


if __name__ == "__main__":
    unittest.main()
