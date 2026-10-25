"""Application workflows shared by the HTTP API and durable job worker."""
import json
from pathlib import Path
import secrets
import uuid

from PIL import Image

import avatar_enhance
import app_settings
import avatar_gallery
import avatar_gallery_jobs
import avatar_guide
import avatar_identity as identity
import avatar_quality as quality
import avatar_benchmark as benchmark
import avatar_content as content
import avatar_face
import image_review
import enhancement_analysis
import avatar_studio as studio
import generation_service as generation
import model_manager
from web_jobs import Cancelled
from chat_store import ChatStore


class StudioService:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.chats = ChatStore(self.root)

    def file(self, value):
        candidate = (self.root / value).resolve()
        if not candidate.is_relative_to(self.root) or not candidate.is_file():
            raise ValueError("The requested image is not in this workspace")
        return candidate

    def relative(self, path):
        return Path(path).resolve().relative_to(self.root).as_posix()

    def images(self, paths):
        result = []
        for path in paths:
            with Image.open(self.file(path)) as image:
                result.append(image.convert("RGB"))
        return result

    def validate(self, image, avatar, prompt, source=None):
        avatar = identity.freeze(str(self.root), avatar)
        if model_manager.DEV_MODE:
            result = studio.validate_generated_image(image, avatar, prompt, dev_mode=True)
            return {**result, "identity_version": avatar.get("identity_version"), "mock_not_quality_evidence": True}
        app_settings.prepare_vision(self.root)
        return quality.validate(str(self.root), image, avatar, prompt, source=source)

    def ready_avatar(self, avatar_id):
        avatar = studio.load_avatar(str(self.root), avatar_id)
        if not avatar.get("face_locked") or not avatar.get("body_locked"):
            raise ValueError("Confirm the face and body first")
        return identity.freeze(str(self.root), avatar)

    def job_avatar(self, args, context, version=None):
        avatar = self.ready_avatar(args["avatar_id"])
        saved_version = context.saved.get("identity_version")
        target = version or saved_version or avatar["identity_version"]
        avatar = identity.pin(str(self.root), avatar, target)
        saved_profile = context.saved.get("quality_profile") if saved_version == target else None
        if saved_profile is None or saved_profile.get("identity_version", target) != target:
            saved_profile = quality.profile(str(self.root), avatar)
        saved_profile = {**saved_profile, "identity_version": target}
        context.checkpoint(identity_version=target, quality_profile=saved_profile)
        return {**avatar, "_quality_profile": saved_profile}

    def references(self, avatar, prompt="", extras=None, face_view="auto", body_view="auto"):
        paths = identity.select_references(avatar, prompt, face_view, body_view)
        return self.images([self.relative(path) for path in paths]) + list(extras or [])[:2]

    def prepare_generation(self):
        if not model_manager.DEV_MODE:
            app_settings.prepare_generation(self.root)

    def _native(self, prompt, refs, aspect, context):
        context.check()
        self.prepare_generation()
        image, seed = generation.generate(model_manager.FLUX_MODEL_NAME, prompt, aspect=aspect,
                                           references=refs, canvas=None, steps=4, on_status=context.emit)
        context.check()
        return image, seed

    def reference(self, kind, details, avatar, context):
        context.emit(f"Generating {kind} reference")
        prompt = studio.build_reference_prompt(kind, details, avatar)
        if model_manager.DEV_MODE:
            return studio.make_dev_reference(kind, details)
        refs = self.images([self.relative(avatar["face_image"])]) if kind == "body" and avatar.get("face_locked") else []
        image, _ = self._native(prompt, refs, "1024x1024 (1:1)" if kind == "face" else "864x1152 (3:4)", context)
        return image

    def main_history(self, chat_id=None):
        return self.chats.get(chat_id)

    def save_main(self, history):
        self.chats.save(history)
        # Keep the legacy entry point and its archive action compatible.
        studio._atomic_json(self.root / "_sessions" / "web_chat.json", history)

    def handle(self, kind, args, context):
        if kind == "generate":
            return self.main_generate(args, context)
        avatar_id = args.get("avatar_id")
        avatar = studio.load_avatar(str(self.root), avatar_id) if avatar_id else None
        if kind in {"reference", "confirm"} and args["reference_kind"] == "body" and not avatar.get("face_locked"):
            raise ValueError("Confirm the face before creating or confirming a body reference")
        if kind == "reference":
            image = self.reference(args["reference_kind"], args["details"], avatar, context)
            updated = studio.save_candidate(str(self.root), avatar_id, args["reference_kind"], image, args["details"])
            return {"avatar_id": avatar_id, "candidate": self.relative(updated[args["reference_kind"] + "_candidate"])}
        if kind == "confirm":
            context.emit("Analyzing and confirming identity")
            image = self.images([args["image"]])[0]
            studio.lock_reference(str(self.root), avatar_id, args["reference_kind"], image, dev_mode=model_manager.DEV_MODE)
            return {"avatar_id": avatar_id}
        if kind == "guide":
            return self.guide(args, context)
        if kind == "console":
            return self.console(args, context)
        if kind == "gallery":
            return self.gallery(args, context)
        if kind == "regenerate":
            return self.regenerate(args, context)
        if kind == "enhance":
            return self.enhance(args, context)
        if kind in {"identity_view", "calibrate", "benchmark", "quality_check", "caption", "publish", "reconcile"}:
            return self.influencer(kind, args, context)
        if kind == "mask_preview":
            image = self.images([args["image"]])[0]
            mask = generation.resolve_mask(image, args["mask_mode"], prompt=args.get("prompt", ""))
            if mask is None:
                raise ValueError("Choose an automatic mask mode")
            path = self.root / "_uploads" / f"mask-{context.id}.png"
            Image.fromarray(mask).save(path)
            return {"image": self.relative(path)}
        raise ValueError("Unknown job type")

    def main_generate(self, args, context):
        chat_id = context.saved.get("chat_id") or args.get("chat_id") or self.chats.get()["id"]
        context.checkpoint(chat_id=chat_id)
        history = self.main_history(chat_id)
        if not any(turn.get("id") == context.id + "-user" for turn in history["turns"]):
            history = self.chats.append(chat_id, None, {"id": context.id + "-user", "role": "user", "text": args["prompt"], "images": args.get("images", [])})
            self.save_main(history)
        saved = list(context.saved.get("images", []))
        seed = context.saved.get("seed") or int(args.get("seed", 0)) or secrets.randbelow(2**63)
        context.checkpoint(seed=seed)
        refs = self.images(args.get("images", []))
        for index in range(len(saved), args.get("quantity", 1)):
            context.emit(f"Generating image {index + 1}/{args.get('quantity', 1)}", image=index + 1, total=args.get("quantity", 1))
            self.prepare_generation()
            image, actual = generation.generate(args["model"], args["prompt"], args.get("negative", generation.DEFAULT_NEGATIVE),
                args["aspect"], seed + index, args.get("cfg", 1), args.get("denoise", 1), args["steps"],
                refs, args.get("canvas"), args.get("mask_mode", "None"), self.file(args["mask"]) if args.get("mask") else None, context.emit)
            path = self.root / f"gen-{context.id}-{index}.png"
            image.save(path)
            saved.append(self.relative(path))
            context.checkpoint(images=saved)
        history = self.main_history(chat_id)
        if not any(turn.get("id") == context.id + "-assistant" for turn in history["turns"]):
            history["turns"].append({"id": context.id + "-assistant", "role": "assistant", "text": args["model"], "images": saved, "seed": str(seed), "recipe": args})
            self.save_main(history)
        return {"images": saved, "seed": str(seed), "chat_id": chat_id}

    def console(self, args, context):
        avatar = self.job_avatar(args, context)
        chat_id = context.saved.get("chat_id") or args.get("chat_id") or self.chats.get(avatar_id=avatar["id"])["id"]
        self.chats.get(chat_id, avatar["id"])
        context.checkpoint(chat_id=chat_id)
        self.chats.append(chat_id, avatar["id"], {"id": context.id + "-user", "role": "user", "text": args["prompt"], "images": args.get("images", [])})
        existing = next((turn for turn in avatar.get("chat_history", []) if turn.get("role") == "assistant"
                         and (turn.get("metadata") or {}).get("job_id") == context.id and turn.get("image_path")), None)
        if existing:
            self.chats.append(chat_id, avatar["id"], {"id": context.id + "-assistant", "role": "assistant", "text": existing.get("content", ""), "images": [self.relative(existing["image_path"])], "seed": (existing.get("metadata") or {}).get("seed")})
            studio.sync_console_gallery(str(self.root), avatar["id"])
            return {"avatar_id": avatar["id"], "image": self.relative(existing["image_path"])}
        if context.saved.get("image"):
            return {"avatar_id": avatar["id"], "image": context.saved["image"]}
        extras = self.images(args.get("images", []))
        paths = context.saved.get("extra_paths")
        if paths is None:
            paths = studio.save_console_references(str(self.root), avatar["id"], extras)
            if not any(turn.get("role") == "user" and (turn.get("metadata") or {}).get("job_id") == context.id for turn in avatar.get("chat_history", [])):
                avatar = studio.append_avatar_chat(str(self.root), avatar["id"], "user", args["prompt"], metadata={"extra_references": paths, "mode": args["mode"], "job_id": context.id, "chat_id": chat_id})
            context.checkpoint(extra_paths=paths)
        avatar = identity.pin(str(self.root), avatar, context.saved["identity_version"])
        avatar["_quality_profile"] = context.saved["quality_profile"]
        prompt = studio.build_generation_prompt(avatar, args["prompt"], args["mode"])
        roles = args.get("roles") or [dict(zip(["Outfit Focus", "Pose Focus", "Scene Focus", "Group Image"], identity.ROLES)).get(args["mode"], "Outfit")] * len(extras)
        prompt += " " + identity.role_instructions(roles)
        views = (args.get("face_view", "auto"), args.get("body_view", "auto"))
        refs = self.references(avatar, args["prompt"], extras, *views)
        context.emit("Generating avatar image")
        if context.saved.get("generated_path"):
            path, seed = context.saved["generated_path"], context.saved["seed"]
            with Image.open(path) as saved_image:
                image = saved_image.convert("RGB")
        else:
            self.prepare_generation()
            image, seed = generation.generate(model_manager.FLUX_MODEL_NAME, prompt, aspect=args["aspect"],
                                              references=refs, steps=args.get("steps", 4), on_status=context.emit)
            path = studio.save_chat_image(str(self.root), avatar["id"], image)
            context.checkpoint(generated_path=path, seed=seed)
        context.check()
        validation = self.validate(image, avatar, prompt)
        recipe = {**identity.recipe(avatar, identity.select_references(avatar, args["prompt"], *views) + paths, roles), "face_view": views[0], "body_view": views[1], "prompt": prompt, "seed": seed, "steps": args.get("steps", 4), "aspect": args["aspect"]}
        studio.append_avatar_chat(str(self.root), avatar["id"], "assistant", f"Generated with {args['mode']}. Seed {seed}.", image_path=path,
            metadata={"prompt": prompt, "seed": seed, "aspect": args["aspect"], "mode": args["mode"], "extra_references": paths, "job_id": context.id, "chat_id": chat_id,
                      "identity_version": avatar["identity_version"], "recipe": recipe, "validation": validation})
        self.chats.append(chat_id, avatar["id"], {"id": context.id + "-assistant", "role": "assistant", "text": f"Generated with {args['mode']}.", "images": [self.relative(path)], "seed": seed})
        studio.sync_console_gallery(str(self.root), avatar["id"])
        item = next((row for row in studio.load_gallery_items(str(self.root), avatar["id"]) if row.get("generated_path") == path), None)
        if item:
            studio.update_gallery_item(str(self.root), avatar["id"], item["id"], validation=validation, recipe=recipe,
                identity_version=avatar["identity_version"], status="passed" if validation.get("pass") else "needs_review")
        context.checkpoint(image=self.relative(path))
        return {"avatar_id": avatar["id"], "image": self.relative(path)}

    def guide(self, args, context):
        avatar_id = args["avatar_id"]
        if args.get("options"):
            options = args["options"]
            studio_avatar = studio.load_avatar(str(self.root), avatar_id)
            previous = avatar_guide.options(studio_avatar)
            avatar_guide.save_options(str(self.root), avatar_id, options.get("face", previous[0]), options.get("body", previous[1]),
                                      options.get("scene_one", previous[2]), options.get("scene_two", previous[3]))
        uploaded = self.images([args["image"]])[0] if args.get("image") else None
        def make(kind, details, avatar):
            context.check()
            if kind != "starter":
                return self.reference(kind, details, avatar, context)
            avatar = identity.freeze(str(self.root), avatar)
            refs = self.references(avatar, details)
            return self._native(details, refs, "864x1152 (3:4)", context)[0]
        for _, message in avatar_guide.advance(str(self.root), avatar_id, uploaded, make, self.validate, dev_mode=model_manager.DEV_MODE):
            context.emit(message)
            context.checkpoint(stage=avatar_guide.stage(studio.load_avatar(str(self.root), avatar_id)))
        return {"avatar_id": avatar_id}

    def mock_references(self, avatar_id, quantity, theme):
        rows = []
        for index in range(min(20, int(quantity))):
            image = studio.make_dev_reference("body", f"{theme} #{index + 1}")
            path = studio.save_gallery_reference(str(self.root), avatar_id, image, f"mock-{uuid.uuid4().hex[:8]}")
            rows.append({"path": path, "image_url": f"dev://{uuid.uuid4().hex}", "description": theme, "source_url": "dev://preview"})
        return rows, {"selected": len(rows)}

    def gallery_prompt(self, avatar, theme, reference):
        if model_manager.DEV_MODE:
            return studio.build_generation_prompt(avatar, theme or "Editorial photograph", "Outfit Focus")
        return avatar_gallery.create_generation_prompt(avatar["name"], theme, reference,
            studio.summarize_specs(avatar.get("face_specs"), limit=10), studio.summarize_specs(avatar.get("body_specs"), limit=10))

    def gallery_image(self, avatar, reference, prompt, aspect, item_id, attempt, context, extras=None, identity_only=False):
        avatar = identity.freeze(str(self.root), avatar)
        previous = studio.gallery_item_by_id(str(self.root), avatar["id"], item_id) or {}
        views = tuple((previous.get("recipe") or {}).get(key, "auto") for key in ("face_view", "body_view")) if identity_only else ("auto", "auto")
        refs = self.references(avatar, prompt, extras, *views)
        if not identity_only:
            if not reference.get("path") or not Path(reference["path"]).is_file():
                raise ValueError("The saved reference image is missing")
            with Image.open(reference["path"]) as image:
                refs = self.references(avatar, prompt, [image.convert("RGB")])
        image, seed = self._native(prompt, refs, aspect, context)
        path = studio.save_gallery_image(str(self.root), avatar["id"], item_id, image, attempt)
        extra_paths = ([reference["path"]] if not identity_only else (studio.gallery_item_by_id(str(self.root), avatar["id"], item_id) or {}).get("extra_references", []))
        studio.update_gallery_item(str(self.root), avatar["id"], item_id, generated_path=path, seed=seed, status="checking", attempt=attempt,
            identity_version=avatar["identity_version"], recipe={**identity.recipe(avatar, identity.select_references(avatar, prompt, *views) + extra_paths), "face_view": views[0], "body_view": views[1], "prompt": prompt, "seed": seed, "aspect": aspect, "steps": 4})
        return path, None, seed

    def gallery(self, args, context):
        self.ready_avatar(args["avatar_id"])
        def make(*items):
            return self.gallery_image(*items, context)
        repair = (lambda prompt, result: prompt + " Correct identity and anatomy.") if model_manager.DEV_MODE else avatar_gallery.repair_generation_prompt
        iterator = avatar_gallery_jobs.run(str(self.root), args["avatar_id"], args["quantity"], args["aspect"], args["theme"],
            args.get("require_face", False), self.gallery_prompt, make, self.validate, repair,
            mock_discover=self.mock_references if model_manager.DEV_MODE else None,
            batch_id=context.saved.get("batch_id") or args.get("batch_id") or (context.id[:12] if args.get("new_batch") else None),
            new_batch=args.get("new_batch", False), name=args.get("batch_name", ""))
        try:
            previous_phase = None
            for row in iterator:
                if context.store.get(context.id).get("cancel_requested"):
                    avatar_gallery_jobs.stop_batch(str(self.root), args["avatar_id"], row["id"])
                context.emit(row["message"], phase=row["phase"], batch_id=row["id"])
                marker = (row["phase"], row.get("updated_at"), row["message"])
                if marker != previous_phase:
                    context.checkpoint(batch_id=row["id"], phase=row["phase"])
                    previous_phase = marker
        finally:
            iterator.close()
        return {"avatar_id": args["avatar_id"]}

    def regenerate(self, args, context):
        item = studio.gallery_item_by_id(str(self.root), args["avatar_id"], args["item_id"])
        if not item or item.get("origin") in {"legacy", "enhancement"} or not item.get("prompt"):
            raise ValueError("This image has no reusable generation recipe")
        avatar = self.job_avatar(args, context, item.get("identity_version"))
        if not context.saved.get("rejected"):
            item = studio.reject_gallery_item(str(self.root), avatar["id"], item["id"], args.get("reason", ""))
            context.checkpoint(rejected=True)
        prompt = context.saved.get("prompt")
        if not prompt:
            prompt = item["prompt"] + " Correct: " + (args.get("reason") or "identity and anatomy") if model_manager.DEV_MODE or item.get("origin") in {"console", "guided", "benchmark"} else avatar_gallery.repair_generation_prompt(item["prompt"], item["validation"])
            context.checkpoint(prompt=prompt)
        if context.saved.get("generated_path"):
            path, seed = context.saved["generated_path"], context.saved["seed"]
        else:
            context.emit("Regenerating selected image")
            extra = self.images([self.relative(path) for path in item.get("extra_references", []) if Path(path).is_file()])
            path, _, seed = self.gallery_image(avatar, {"path": item.get("reference_path")}, prompt,
                item.get("aspect", "864x1152 (3:4)"), item["id"], int(item.get("attempt", 1)) + 1, context,
                extras=extra, identity_only=item.get("origin") in {"console", "guided", "benchmark"})
            context.checkpoint(generated_path=path, seed=seed)
        context.emit("Checking regenerated image")
        with Image.open(path) as image:
            validation = self.validate(image.convert("RGB"), avatar, prompt)
        studio.update_gallery_item(str(self.root), avatar["id"], item["id"], prompt=prompt,
            generated_path=path, seed=seed, validation=validation, status="passed" if validation.get("pass") else "needs_review")
        return {"avatar_id": avatar["id"], "item_id": item["id"]}

    def enhance(self, args, context):
        args = {**args, "reconstruction_prompt": args.get("reconstruction_prompt", avatar_enhance.PHOTO_PROMPT)}
        avatar_id, source_id = args.get("avatar_id"), args.get("item_id")
        chat_id = args.get("chat_id")
        if chat_id:
            conversation = self.chats.get(chat_id, avatar_id)
            if not any(args.get("image") in turn.get("images", []) for turn in conversation["turns"] if turn.get("role") == "assistant"):
                raise ValueError("Select a generated image from this conversation")
            source_path = str(self.file(args["image"]))
        else:
            item = studio.gallery_item_by_id(str(self.root), avatar_id, source_id)
            if not item:
                raise ValueError("The selected image no longer exists")
            source_path = item["generated_path"]
        item_id = f"enhance-{context.id}"
        with Image.open(source_path) as original:
            source = original.convert("RGB")
        source, source_hash = enhancement_analysis.fingerprint(source)
        plan = context.saved.get("enhancement_plan")
        if plan and plan.get("source_sha256") != source_hash:
            raise ValueError("The source image changed since this enhancement started. Start a new enhancement.")
        if not plan:
            if args.get("adaptive_prompt", True) and not context.saved.get("image"):
                profile = args.get("enhancement_profile", "auto")
                if profile == "auto":
                    context.emit("Analyzing source for an image-aware reconstruction prompt")
                    if not model_manager.DEV_MODE:
                        app_settings.prepare_vision(self.root)
                    analysis = enhancement_analysis.analyze(source, self.root, mock=model_manager.DEV_MODE,
                        check=context.check, emit=context.emit)
                else:
                    analysis = {"status": "disabled", "reason": "manual_profile"}
                instructions = args.get("additional_instructions", "").strip() or args["reconstruction_prompt"]
                plan = enhancement_analysis.plan(analysis, instructions, profile)
            else:
                plan = {"automatic": False, "selection": "custom", "analysis": {"status": "legacy_unrecorded" if context.saved.get("image") else "disabled"}, "prompt": args["reconstruction_prompt"]}
            plan.update(source_sha256=source_hash, requested_prompt=plan.get("user_instructions", args["reconstruction_prompt"]),
                        additional_instructions=args.get("additional_instructions", ""))
            context.checkpoint(enhancement_plan=plan)
        sampling_mode = args.get("sampling_mode", "img2img")
        args = {**args, "reconstruction_prompt": plan["prompt"]}

        def review_output():
            if context.saved.get("texture_done"):
                return
            diagnostics = dict(context.saved.get("diagnostics") or {})
            diagnostics["enhancement_plan"] = plan
            if plan["automatic"]:
                context.emit("Checking reconstruction scene, focus and visible artifacts")
                if not model_manager.DEV_MODE:
                    app_settings.prepare_vision(self.root)
                result = self.images([context.saved["image"]])[0]
                diagnostics["texture_review"] = enhancement_analysis.review_texture(result, plan["analysis"],
                    mock=model_manager.DEV_MODE, check=context.check)
                if plan.get("version") == enhancement_analysis.VERSION:
                    diagnostics["content_review"] = diagnostics["texture_review"]
            else:
                diagnostics["texture_review"] = {"status": "disabled", "reasons": []}
            context.checkpoint(diagnostics=diagnostics, texture_done=True)
            if avatar_id:
                studio.update_gallery_item(str(self.root), avatar_id, item_id, diagnostics=diagnostics,
                    status=enhancement_analysis.output_status(diagnostics))

        def check_output():
            if not avatar_id or context.saved.get("quality_done"):
                return
            item = studio.gallery_item_by_id(str(self.root), avatar_id, source_id) if source_id else None
            if not item:
                item = next((row for row in studio.load_gallery_items(str(self.root), avatar_id) if row.get("generated_path") == source_path), None)
            avatar = self.job_avatar(args, context, item.get("identity_version") if item else None)
            context.emit("Checking reconstruction identity and source preservation")
            image = self.images([context.saved["image"]])[0]
            original = self.images([self.relative(source_path)])[0]
            validation = self.validate(image, avatar, args["reconstruction_prompt"], original)
            current = studio.gallery_item_by_id(str(self.root), avatar_id, item_id)
            status = enhancement_analysis.output_status(current.get("diagnostics"), validation.get("pass"))
            studio.update_gallery_item(str(self.root), avatar_id, item_id, validation=validation, status=status,
                identity_version=avatar["identity_version"], recipe=identity.recipe(avatar))
            context.checkpoint(quality_done=True)
        def save_turn(image):
            if chat_id:
                self.chats.append(chat_id, avatar_id, {"id": context.id + "-assistant", "role": "assistant",
                    "text": "Enhanced image: needs review" if enhancement_analysis.needs_review(context.saved.get("diagnostics")) else "Enhanced image", "images": [image], "source": self.relative(source_path),
                    "reconstruction_prompt": args["reconstruction_prompt"],
                    "diagnostics": context.saved.get("diagnostics"),
                    "seed": str(context.saved.get("seed", "")), "processing_passes": context.saved.get("processing_passes")})
        if context.saved.get("image"):
            review_output()
            check_output()
            save_turn(context.saved["image"])
            return {"avatar_id": avatar_id, "item_id": item_id, "image": context.saved["image"]}
        seed = context.saved.get("seed")
        if seed is None:
            seed = secrets.randbelow(2**63)
        context.checkpoint(seed=seed)
        if avatar_id:
            parent = next((row for row in studio.load_gallery_items(str(self.root), avatar_id)
                           if row.get("generated_path") == source_path), {})
            studio.record_gallery_item(str(self.root), avatar_id, {"id": item_id, "origin": "enhancement", "source_id": source_id,
                "batch_id": parent.get("batch_id"),
                "source_path": source_path, "status": "enhancing", "attempt": 1, "prompt": args["reconstruction_prompt"],
                "method": args["method"], "strength": args["strength"], "steps": args["steps"], "scale": args["scale"], "seed": str(seed)})
        def edit(tile, whole, prompt, seed, strength):
            context.check()
            self.prepare_generation()
            engine = model_manager.ensure_model(model_manager.FLUX_MODEL_NAME, status_callback=context.emit)
            if sampling_mode == "reference":
                context.emit("FLUX reference reconstruction / native Flux2Scheduler / four full-schedule steps")
                return engine.reconstruct_reference([tile] if whole is None else [tile, whole], prompt, seed, args["steps"])
            return engine.img2img([tile] if whole is None else [tile, whole], prompt, "", seed, 1.0, strength, args["steps"])
        try:
            for row in avatar_enhance.reconstruct(source, edit, args["scale"], args["strength"], seed,
                    lambda: context.store.get(context.id).get("cancel_requested", False), args["method"], args["reconstruction_prompt"], preserve_tone=args.get("preserve_tone", True)):
                if row["image"] is None:
                    context.emit(f"FLUX pass {row['tile']}/{row['total']} / one output image", pass_index=row["tile"], total=row["total"])
                else:
                    row["diagnostics"]["sampling"] = {
                        "mode": sampling_mode,
                        "scheduler": "Flux2Scheduler" if sampling_mode == "reference" else "simple",
                        "steps": args["steps"],
                        "denoise": 1.0 if sampling_mode == "reference" else args["strength"],
                        "latent": "EmptyFlux2LatentImage" if sampling_mode == "reference" else "VAEEncode",
                        "output_size_mode": "one sampled image then resize" if args["method"] == "photo" else "full image and sampled HD tiles",
                        "tile_prompt_prefix": avatar_enhance.TILE_PROMPT if args["method"] == "hd_tiles" else None,
                    }
                    if avatar_id:
                        path = studio.save_gallery_image(str(self.root), avatar_id, item_id, row["image"])
                        if not model_manager.DEV_MODE:
                            try:
                                row["diagnostics"]["detail_regions"] = image_review.regions(source.size, avatar_face.features(source))
                            except Exception:
                                pass  # Center-detail comparison remains available without face detection.
                        studio.update_gallery_item(str(self.root), avatar_id, item_id, generated_path=path,
                            status="no_change" if row["diagnostics"]["near_noop"] else "unchecked", diagnostics=row["diagnostics"],
                            processing_passes=row["total"], working_size=list(row["working_size"]), output_size=list(row["size"]))
                    else:
                        path = self.root / (item_id + ".png")
                        row["image"].save(path)
                    context.checkpoint(image=self.relative(path), processing_passes=row["total"], diagnostics=row["diagnostics"])
                    review_output()
                    check_output()
                    save_turn(self.relative(path))
        except avatar_enhance.EnhancementCancelled as exc:
            if avatar_id:
                studio.update_gallery_item(str(self.root), avatar_id, item_id, status="stopped")
            raise Cancelled(str(exc)) from exc
        return {"avatar_id": avatar_id, "item_id": item_id, "image": context.saved.get("image")}

    def influencer(self, kind, args, context):
        selected = studio.gallery_item_by_id(str(self.root), args["avatar_id"], args["item_id"]) if kind == "quality_check" else None
        avatar = self.job_avatar(args, context, selected.get("identity_version") if selected else None)
        root, avatar_id = str(self.root), avatar["id"]
        if kind == "identity_view":
            candidate_id = context.id[:12]
            if context.saved.get("candidate_id") or any(row["id"] == candidate_id for row in identity.pack(root, avatar_id)["candidates"]):
                return {"candidate_id": candidate_id}
            view = args["view"]
            prompt = studio.build_generation_prompt(avatar,
                f"One clean identity reference photograph, {view.replace('-', ' ')} view. "
                "Same adult person as the identity references; preserve face and body proportions. "
                + ("Entire body and feet visible." if view.startswith("body") else "Tight head-and-neck portrait, entire hairstyle visible.")
                + " Plain studio background, even lighting, natural skin texture. " + args.get("details", ""))
            image, _ = self._native(prompt, self.references(avatar, prompt), "864x1152 (3:4)" if view.startswith("body") else "1024x1024 (1:1)", context)
            candidate_id = identity.add_candidate(root, avatar_id, view, image, candidate_id)
            context.checkpoint(candidate_id=candidate_id)
            return {"candidate_id": candidate_id}
        if kind == "calibrate":
            positives, negatives = args["positives"], args["negatives"]
            if len(positives) < 2 or len(negatives) < 2:
                raise ValueError("Select at least two same-person and two different-person calibration photographs.")
            hashes = [identity._digest(self.file(path)) for path in positives + negatives]
            if len(set(hashes)) != len(hashes):
                raise ValueError("Calibration photographs must be distinct, with no overlap between groups.")
            context.emit("Calibrating CPU face comparison")
            face = {"calibrated": True, "validation_passed": True, "threshold": .5, "margin": .04, "mock_not_quality_evidence": True} if model_manager.DEV_MODE else avatar_face.calibrate(
                [avatar["face_image"]] + [row["path"] for row in avatar.get("_identity_pack", []) if not row["view"].startswith("body")], [str(self.file(path)) for path in positives], [str(self.file(path)) for path in negatives])
            face.update(positive_files=positives, negative_files=negatives)
            quality.save_profile(root, avatar, {"face": face})
            return {"calibrated": face["calibrated"], "mock": model_manager.DEV_MODE}
        if kind == "quality_check":
            item = studio.gallery_item_by_id(root, avatar_id, args["item_id"])
            if not item:
                raise ValueError("Saved image not found.")
            context.emit("Checking identity, appearance and image quality")
            image = self.images([self.relative(item["generated_path"])])[0]
            source = self.images([self.relative(item["source_path"])])[0] if item.get("source_path") else None
            validation = self.validate(image, avatar, item.get("prompt", ""), source)
            status = (enhancement_analysis.output_status(item.get("diagnostics"), validation.get("pass"))
                      if item.get("origin") == "enhancement" else "passed" if validation.get("pass") else "needs_review")
            studio.update_gallery_item(root, avatar_id, item["id"], validation=validation,
                identity_version=avatar["identity_version"], status=status)
            return {"item_id": item["id"], "validation": validation}
        if kind == "benchmark":
            def make(pinned, scene, seed):
                context.check()
                self.prepare_generation()
                return generation.generate(model_manager.FLUX_MODEL_NAME, studio.build_generation_prompt(pinned, scene),
                    references=self.references(pinned, scene), seed=seed, steps=4, aspect="864x1152 (3:4)", on_status=context.emit)[0]
            iterator = benchmark.run(root, avatar_id, make, self.validate, model_manager.DEV_MODE, context.check, context.id)
            try:
                for row in iterator:
                    context.emit(f"Consistency test: {benchmark.report(row)['completed']}/{len(row['cases'])}")
                    context.checkpoint(benchmark_id=row["id"])
            finally:
                iterator.close()
            return benchmark.report(benchmark.load(root, avatar_id))
        if kind == "caption":
            text = context.saved.get("caption")
            if text is None:
                text = content.caption(root, avatar_id, args["item_id"], args.get("details", ""), model_manager.DEV_MODE, args.get("use_api", False), args.get("caption_style", "auto"))
                context.checkpoint(caption=text)
            return {"caption": text, "source": "gemini" if args.get("use_api") else "local_templates"}
        if kind == "publish":
            return {"post": content.publish(root, avatar_id, args["post_id"], model_manager.DEV_MODE)}
        if kind == "reconcile":
            if model_manager.DEV_MODE:
                raise ValueError("Instagram calls are disabled in local preview.")
            return {"post": content.reconcile(root, avatar_id, args["post_id"], args["media_id"])}
        raise ValueError("Unknown influencer operation.")
