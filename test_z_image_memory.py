import sys
import types
import unittest
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


if __name__ == "__main__":
    unittest.main()
