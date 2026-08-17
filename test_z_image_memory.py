import ast
import base64
import html
import json
import os
import re
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
import avatar_gallery
import avatar_studio


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

        calls = []

        def fake_conditioning_set_values(conditioning, values, append=False):
            calls.append((values, append))
            return conditioning + values["reference_latents"]

        nodes = {
            "VAEEncode": FakeVaeEncode(),
            "conditioning_set_values": fake_conditioning_set_values,
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
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(append for _values, append in calls))
        for latent_samples in conditioned:
            self.assertLessEqual(latent_samples.width * latent_samples.height, 1024**2 // 3)

    def test_flux_reference_limit_is_four(self):
        images = [Image.new("RGB", (32, 32)) for _ in range(5)]
        with self.assertRaisesRegex(ValueError, "at most 4"):
            engine_flux_klein_4b._normalize_references(images)

    def test_attachment_queue_appends_and_tracks_canvas_state(self):
        namespace = {"MAX_FLUX_REFERENCES": 4}
        namespace["_as_image_list"] = _source_function("app.py", "_as_image_list", namespace)
        append_images = _source_function("app.py", "_append_attachment_images", namespace)
        apply_action = _source_function("app.py", "_apply_attachment_action", namespace)
        images = [Image.new("RGB", (8, 8), color) for color in ("red", "blue", "green")]

        queue = append_images([images[0]], images[1:])
        self.assertEqual(queue, images)
        self.assertEqual(apply_action(queue, 2, "canvas"), (images, 2))
        self.assertEqual(apply_action(queue, 2, "canvas", 2), (images, None))
        self.assertEqual(apply_action(queue, 1, "left"), ([images[1], images[0], images[2]], 1))
        self.assertEqual(apply_action(queue, 1, "right"), ([images[0], images[2], images[1]], 0))
        self.assertEqual(apply_action(queue, 1, "remove"), ([images[0], images[2]], 0))
        with self.assertRaisesRegex(ValueError, "up to 4"):
            append_images(queue, [images[0], images[1]])

    def test_generated_gallery_item_can_be_added_from_gradio_file_data(self):
        converter = _source_function(
            "app.py", "_gallery_item_to_pil", {"Image": Image, "os": os}
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "result.png")
            Image.new("RGB", (17, 19), "purple").save(path)
            restored = converter({"image": {"path": path}})
        self.assertEqual(restored.size, (17, 19))

    def test_chat_history_round_trips_through_persistent_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            namespace = {
                "CHAT_HISTORY_PATH": os.path.join(temp_dir, "sessions", "current_chat.json"),
                "datetime": __import__("datetime").datetime,
                "json": __import__("json"),
                "os": os,
                "uuid": __import__("uuid"),
            }
            save_history = _source_function("app.py", "_save_chat_history", namespace)
            load_history = _source_function("app.py", "_load_chat_history", namespace)
            turns = ["<div>You</div>", "<div>FreeFakeStudio</div>"]
            save_history(turns)
            self.assertEqual(load_history(), turns)


class AvatarStudioTests(unittest.TestCase):
    def test_avatar_create_lock_and_restore_specs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            avatar = avatar_studio.create_avatar(temp_dir, "Maya Test")
            self.assertEqual(avatar["current_step"], "face")
            self.assertFalse(avatar["face_locked"])

            face = avatar_studio.make_dev_reference("face", "dark hair")
            avatar, face_specs = avatar_studio.lock_reference(
                temp_dir, avatar["id"], "face", face, dev_mode=True
            )
            self.assertTrue(avatar["face_locked"])
            self.assertEqual(avatar["current_step"], "body")
            self.assertTrue(os.path.exists(avatar["face_image"]))
            self.assertEqual(face_specs["analyzer"], "mock-smolvlm-dev")

            body = avatar_studio.make_dev_reference("body", "full body")
            avatar, body_specs = avatar_studio.lock_reference(
                temp_dir, avatar["id"], "body", body, dev_mode=True
            )
            self.assertTrue(avatar["body_locked"])
            self.assertEqual(avatar["current_step"], "console")
            self.assertTrue(os.path.exists(avatar["body_image"]))
            self.assertIn("body_build", avatar_studio.summarize_specs(body_specs))

            restored = avatar_studio.selected_or_first(temp_dir)
            self.assertEqual(restored["id"], avatar["id"])

    def test_avatar_prompt_uses_locked_specs_and_caps_extra_references(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            avatar = avatar_studio.create_avatar(temp_dir, "Reference Slots")
            avatar, _ = avatar_studio.lock_reference(
                temp_dir, avatar["id"], "face", avatar_studio.make_dev_reference("face"), dev_mode=True
            )
            avatar, _ = avatar_studio.lock_reference(
                temp_dir, avatar["id"], "body", avatar_studio.make_dev_reference("body"), dev_mode=True
            )
            extras = [
                avatar_studio.make_dev_reference("body", "extra 1"),
                avatar_studio.make_dev_reference("body", "extra 2"),
                avatar_studio.make_dev_reference("body", "extra 3"),
            ]

            references = avatar_studio.reference_images(temp_dir, avatar, extras)
            self.assertEqual(len(references), 4)

            prompt = avatar_studio.build_generation_prompt(
                avatar, "standing in a studio campaign", "Outfit Focus"
            )
            self.assertIn("Use extra references for outfit", prompt)
            self.assertIn("standing in a studio campaign", prompt)

    def test_avatar_non_dev_uses_vision_analyzer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            avatar = avatar_studio.create_avatar(temp_dir, "Vision Path")
            fake_vision = types.ModuleType("avatar_vision")
            calls = []

            def fake_analyze(kind, image, questions):
                calls.append((kind, image.size, len(questions)))
                return {
                    "kind": kind,
                    "analyzer": "fake-smolvlm",
                    "image_size": list(image.size),
                    "answers": [
                        {
                            "key": questions[0][0],
                            "question": questions[0][1],
                            "answer": "yes",
                            "confidence": 0.9,
                            "strictness": questions[0][2],
                        }
                    ],
                }

            fake_vision.analyze_reference_image = fake_analyze
            with mock.patch.dict(sys.modules, {"avatar_vision": fake_vision}):
                avatar, specs = avatar_studio.lock_reference(
                    temp_dir,
                    avatar["id"],
                    "face",
                    avatar_studio.make_dev_reference("face"),
                    dev_mode=False,
                )

            self.assertEqual(specs["analyzer"], "fake-smolvlm")
            self.assertEqual(calls[0][0], "face")
            self.assertTrue(avatar["face_locked"])

    def test_avatar_strict_analyzer_errors_are_not_silently_mocked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            avatar = avatar_studio.create_avatar(temp_dir, "Strict Vision")
            fake_vision = types.ModuleType("avatar_vision")
            fake_vision.analyze_reference_image = mock.Mock(side_effect=RuntimeError("boom"))

            with mock.patch.dict(sys.modules, {"avatar_vision": fake_vision}):
                with self.assertRaises(RuntimeError):
                    avatar_studio.lock_reference(
                        temp_dir,
                        avatar["id"],
                        "face",
                        avatar_studio.make_dev_reference("face"),
                        dev_mode=False,
                    )

    def test_mock_engine_supports_reference_generation(self):
        engine = model_manager.MockEngine(model_manager.FLUX_MODEL_NAME)
        image = engine.generate_with_references([], "prompt", "", 320, 240, 1, 1.0, 1.0, 4)
        self.assertEqual(image.size, (320, 240))

    def test_avatar_gallery_manifest_persists_validation_and_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            avatar = avatar_studio.create_avatar(temp_dir, "Gallery State")
            image = avatar_studio.make_dev_reference("body", "gallery result")
            item_id = "gallery-test"
            image_path = avatar_studio.save_gallery_image(
                temp_dir, avatar["id"], item_id, image, attempt=1
            )
            avatar_studio.record_gallery_item(
                temp_dir,
                avatar["id"],
                {
                    "id": item_id,
                    "status": "generating",
                    "prompt": "studio portrait",
                    "generated_path": image_path,
                },
            )
            avatar_studio.update_gallery_item(
                temp_dir,
                avatar["id"],
                item_id,
                status="passed",
                attempt=1,
                validation={"pass": True, "score": 94},
            )

            items = avatar_studio.load_gallery_items(temp_dir, avatar["id"])
            restored = avatar_studio.load_avatar(temp_dir, avatar["id"])
            self.assertEqual(items[0]["status"], "passed")
            self.assertEqual(items[0]["validation"]["score"], 94)
            self.assertEqual(restored["gallery_count"], 1)

    def test_dev_gallery_validation_never_loads_vision_model(self):
        image = avatar_studio.make_dev_reference("body")
        validation = avatar_studio.validate_generated_image(
            image,
            {"face_specs": None, "body_specs": None},
            "test prompt",
            dev_mode=True,
        )
        self.assertTrue(validation["pass"])
        self.assertEqual(validation["validator"], "mock-smolvlm-dev")

    def test_tavily_candidate_parser_deduplicates_urls(self):
        payload = {
            "images": [{"url": "https://example.com/a.jpg", "description": "top"}],
            "results": [
                {
                    "url": "https://example.com/page",
                    "title": "Page",
                    "images": [
                        "https://example.com/a.jpg",
                        {"url": "https://example.com/b.jpg", "description": "second"},
                    ],
                }
            ],
        }
        candidates = avatar_gallery._candidate_images(payload)
        self.assertEqual([item["image_url"] for item in candidates], [
            "https://example.com/a.jpg",
            "https://example.com/b.jpg",
        ])
        self.assertEqual(candidates[1]["source_url"], "https://example.com/page")

    def test_tavily_search_uses_form_configured_constraints(self):
        captured = {}

        def fake_post(url, headers, json, timeout):
            captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
            return types.SimpleNamespace(ok=True, json=lambda: {"results": [], "images": []})

        env = {
            "TAVILY_API_KEY": "tvly-test",
            "FFS_AVATAR_REFERENCE_DOMAINS": "instagram.com, vogue.com",
            "FFS_AVATAR_REFERENCE_TIME_RANGE": "month",
            "FFS_AVATAR_SAFE_SEARCH": "0",
        }
        with mock.patch.dict(os.environ, env, clear=False), \
             mock.patch.object(avatar_gallery.requests, "post", side_effect=fake_post):
            avatar_gallery._tavily_search("fashion reference", max_results=99)

        self.assertEqual(captured["url"], "https://api.tavily.com/search")
        self.assertEqual(captured["json"]["max_results"], 20)
        self.assertEqual(captured["json"]["auto_parameters"], False)
        self.assertEqual(captured["json"]["time_range"], "month")
        self.assertEqual(captured["json"]["include_domains"], ["instagram.com", "vogue.com"])
        self.assertEqual(captured["json"]["safe_search"], False)

    def test_gemini_request_sends_api_key_header(self):
        captured = {}

        def fake_post(url, params, headers, json, timeout):
            captured.update({"url": url, "params": params, "headers": headers, "json": json})
            return types.SimpleNamespace(
                ok=True,
                json=lambda: {
                    "candidates": [
                        {"content": {"parts": [{"text": '{"ok": true}'}]}}
                    ]
                },
            )

        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-test"}, clear=False), \
             mock.patch.object(avatar_gallery, "_gemini_model", return_value="gemini-test-model"), \
             mock.patch.object(avatar_gallery.requests, "post", side_effect=fake_post):
            result = avatar_gallery.gemini_json("return json", schema)

        self.assertTrue(result["ok"])
        self.assertEqual(captured["params"]["key"], "gemini-test")
        self.assertEqual(captured["headers"]["x-goog-api-key"], "gemini-test")

    def test_gallery_search_retries_variations_until_quantity_is_met(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            round_one = [{"image_url": "https://example.com/one.jpg"}]
            round_two = [
                {"image_url": "https://example.com/two.jpg"},
                {"image_url": "https://example.com/three.jpg"},
            ]

            def fake_download(candidate, output_dir):
                return {
                    **candidate,
                    "path": str(Path(output_dir) / "reference.jpg"),
                    "validation_bytes": b"jpeg-bytes",
                }

            approved = [
                {**item, "path": str(Path(temp_dir) / f"{index}.jpg"), "validation_bytes": b"jpeg"}
                for index, item in enumerate(round_two)
            ]
            with mock.patch.object(avatar_gallery, "configuration_status", return_value={"ready": True}), \
                    mock.patch.object(avatar_gallery, "_search_query", side_effect=["query one", "query two"]), \
                    mock.patch.object(avatar_gallery, "_tavily_search", side_effect=[{}, {}]) as search, \
                    mock.patch.object(avatar_gallery, "_candidate_images", side_effect=[round_one, round_two]), \
                    mock.patch.object(avatar_gallery, "_download_candidate", side_effect=fake_download), \
                    mock.patch.object(avatar_gallery, "_validate_references", side_effect=[[], approved]):
                selected, report = avatar_gallery.discover_references("theme", 2, temp_dir)

            self.assertEqual(search.call_count, 2)
            self.assertEqual(len(selected), 2)
            self.assertEqual(report["selected"], 2)
            saved_report = json.loads(next(Path(temp_dir).glob("search_*.json")).read_text(encoding="utf-8"))
            self.assertNotIn("validation_bytes", saved_report["references"][0])
            self.assertIn("settings", saved_report)

    def test_colab_notebook_keeps_tokens_blank_and_exposes_avatar_controls(self):
        notebook = json.loads(Path("FreeFakeStudio.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn('NGROK_AUTH_TOKEN = ""', source)
        self.assertIn('HUGGINGFACE_TOKEN = ""', source)
        self.assertIn('GEMINI_API_KEY = ""', source)
        self.assertIn('TAVILY_API_KEY = ""', source)
        self.assertIn('PUBLIC_ROUTE = "Colab proxy"', source)
        self.assertIn("PRELOAD_FLUX = True", source)
        self.assertIn('os.environ["FFS_PUBLIC_ROUTE"]', source)
        self.assertIn('os.environ["FFS_PRELOAD_FLUX"]', source)
        self.assertIn("key_bool", source)
        self.assertIn("UPLOAD_KEYS_TXT", source)
        self.assertIn("KEYS_TXT_PATH", source)
        self.assertIn("parse_keys_txt", source)
        self.assertIn('"FFS_PUBLIC_ROUTE": "PUBLIC_ROUTE"', source)
        self.assertIn('"FFS_PRELOAD_FLUX": "PRELOAD_FLUX"', source)
        self.assertIn("freefakestudio_keys.txt", source)
        self.assertIn('"stash", "push", "-u"', source)
        self.assertIn("Colab auto-stash before update", source)
        self.assertIn("AVATAR_SEARCH_ROUNDS", source)
        self.assertIn("AVATAR_VISION_MAX_EDGE", source)
        self.assertNotIn("tvly-", source)
        self.assertIsNone(re.search(r"hf_[A-Za-z0-9]{20,}", source))

    def test_keys_template_is_safe_and_lists_supported_fields(self):
        template = Path("FreeFakeStudio.keys.example.txt").read_text(encoding="utf-8")
        ignore = Path(".gitignore").read_text(encoding="utf-8")

        for key in (
            "NGROK_AUTH_TOKEN",
            "PUBLIC_ROUTE",
            "PRELOAD_FLUX",
            "HUGGINGFACE_TOKEN",
            "GEMINI_API_KEY",
            "TAVILY_API_KEY",
            "FLUX_ENCODER",
            "AVATAR_SEARCH_ROUNDS",
            "AVATAR_VISION_MAX_EDGE",
        ):
            self.assertIn(key, template)
        self.assertIn("FreeFakeStudio.keys.txt", ignore)
        self.assertNotIn("tvly-", template)
        self.assertIsNone(re.search(r"hf_[A-Za-z0-9]{20,}", template))

    def test_app_preloads_only_flux_when_requested(self):
        source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn('FFS_PRELOAD_FLUX', source)
        self.assertIn('model_manager.FLUX_MODEL_NAME', source)
        self.assertIn('Preloading FLUX.2-klein 4B', source)
        self.assertNotIn('ensure_model("Z-Image Turbo"', source)
        self.assertNotIn('ensure_model("ERNIE-Image-Turbo"', source)


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

    def test_public_route_defaults_to_colab_proxy(self):
        normalize = self._launch_function("normalize_public_route", {})

        self.assertEqual(normalize("Colab proxy"), "colab_proxy")
        self.assertEqual(normalize("ngrok"), "ngrok")
        self.assertEqual(normalize("Auto"), "auto")
        with self.assertRaisesRegex(RuntimeError, "Unsupported PUBLIC_ROUTE"):
            normalize("random tunnel")

    def test_bool_normalizer_accepts_colab_form_values(self):
        normalize = self._launch_function("normalize_bool", {})

        self.assertTrue(normalize("True"))
        self.assertTrue(normalize("1"))
        self.assertFalse(normalize("False", True))
        self.assertFalse(normalize("0", True))


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
