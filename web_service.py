"""Application workflows shared by the HTTP API and durable job worker."""
import json
from pathlib import Path
import secrets
import uuid

from PIL import Image

import avatar_enhance
import avatar_gallery
import avatar_gallery_jobs
import avatar_guide
import avatar_studio as studio
import generation_service as generation
import model_manager
from web_jobs import Cancelled


class StudioService:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

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

    def validate(self, image, avatar, prompt):
        return studio.validate_generated_image(image, avatar, prompt, dev_mode=model_manager.DEV_MODE)

    def ready_avatar(self, avatar_id):
        avatar = studio.load_avatar(str(self.root), avatar_id)
        if not avatar.get("face_locked") or not avatar.get("body_locked"):
            raise ValueError("Confirm the face and body first")
        return avatar

    def _native(self, prompt, refs, aspect, context):
        context.check()
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

    def main_history(self):
        path = self.root / "_sessions" / "web_chat.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        from legacy_history import migrate
        history = migrate(self.root)
        if history["turns"]:
            self.save_main(history)
        return history

    def save_main(self, history):
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
        history = self.main_history()
        if not any(turn.get("id") == context.id + "-user" for turn in history["turns"]):
            history["turns"].append({"id": context.id + "-user", "role": "user", "text": args["prompt"], "images": args.get("images", [])})
            self.save_main(history)
        saved = list(context.saved.get("images", []))
        seed = context.saved.get("seed") or int(args.get("seed", 0)) or secrets.randbelow(2**63)
        context.checkpoint(seed=seed)
        refs = self.images(args.get("images", []))
        for index in range(len(saved), args.get("quantity", 1)):
            context.emit(f"Generating image {index + 1}/{args.get('quantity', 1)}", image=index + 1, total=args.get("quantity", 1))
            image, actual = generation.generate(args["model"], args["prompt"], args.get("negative", generation.DEFAULT_NEGATIVE),
                args["aspect"], seed + index, args.get("cfg", 1), args.get("denoise", 1), args["steps"],
                refs, args.get("canvas"), args.get("mask_mode", "None"), self.file(args["mask"]) if args.get("mask") else None, context.emit)
            path = self.root / f"gen-{context.id}-{index}.png"
            image.save(path)
            saved.append(self.relative(path))
            context.checkpoint(images=saved)
        history = self.main_history()
        if not any(turn.get("id") == context.id + "-assistant" for turn in history["turns"]):
            history["turns"].append({"id": context.id + "-assistant", "role": "assistant", "text": args["model"], "images": saved, "seed": str(seed), "recipe": args})
            self.save_main(history)
        return {"images": saved, "seed": str(seed)}

    def console(self, args, context):
        avatar = self.ready_avatar(args["avatar_id"])
        existing = next((turn for turn in avatar.get("chat_history", []) if turn.get("role") == "assistant"
                         and (turn.get("metadata") or {}).get("job_id") == context.id and turn.get("image_path")), None)
        if existing:
            studio.sync_console_gallery(str(self.root), avatar["id"])
            return {"avatar_id": avatar["id"], "image": self.relative(existing["image_path"])}
        if context.saved.get("image"):
            return {"avatar_id": avatar["id"], "image": context.saved["image"]}
        extras = self.images(args.get("images", []))
        paths = context.saved.get("extra_paths")
        if paths is None:
            paths = studio.save_console_references(str(self.root), avatar["id"], extras)
            if not any(turn.get("role") == "user" and (turn.get("metadata") or {}).get("job_id") == context.id for turn in avatar.get("chat_history", [])):
                avatar = studio.append_avatar_chat(str(self.root), avatar["id"], "user", args["prompt"], metadata={"extra_references": paths, "mode": args["mode"], "job_id": context.id})
            context.checkpoint(extra_paths=paths)
        prompt = studio.build_generation_prompt(avatar, args["prompt"], args["mode"])
        refs = studio.reference_images(str(self.root), avatar, extras)
        context.emit("Generating avatar image")
        image, seed = generation.generate(model_manager.FLUX_MODEL_NAME, prompt, aspect=args["aspect"],
                                          references=refs, steps=args.get("steps", 4), on_status=context.emit)
        path = studio.save_chat_image(str(self.root), avatar["id"], image)
        studio.append_avatar_chat(str(self.root), avatar["id"], "assistant", f"Generated with {args['mode']}. Seed {seed}.", image_path=path,
            metadata={"prompt": prompt, "seed": seed, "aspect": args["aspect"], "mode": args["mode"], "extra_references": paths, "job_id": context.id})
        studio.sync_console_gallery(str(self.root), avatar["id"])
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
            refs = studio.reference_images(str(self.root), avatar)
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
        refs = studio.reference_images(str(self.root), avatar, extras or [])
        if not identity_only:
            if not reference.get("path") or not Path(reference["path"]).is_file():
                raise ValueError("The saved reference image is missing")
            with Image.open(reference["path"]) as image:
                refs = studio.reference_images(str(self.root), avatar, [image.convert("RGB")])
        image, seed = self._native(prompt, refs, aspect, context)
        path = studio.save_gallery_image(str(self.root), avatar["id"], item_id, image, attempt)
        studio.update_gallery_item(str(self.root), avatar["id"], item_id, generated_path=path, seed=seed, status="checking", attempt=attempt)
        return path, None, seed

    def gallery(self, args, context):
        self.ready_avatar(args["avatar_id"])
        def make(*items):
            return self.gallery_image(*items, context)
        repair = (lambda prompt, result: prompt + " Correct identity and anatomy.") if model_manager.DEV_MODE else avatar_gallery.repair_generation_prompt
        iterator = avatar_gallery_jobs.run(str(self.root), args["avatar_id"], args["quantity"], args["aspect"], args["theme"],
            args.get("require_face", False), self.gallery_prompt, make, self.validate, repair,
            mock_discover=self.mock_references if model_manager.DEV_MODE else None)
        try:
            previous_phase = None
            for row in iterator:
                if context.store.get(context.id).get("cancel_requested"):
                    avatar_gallery_jobs.stop_batch(str(self.root), args["avatar_id"])
                context.emit(row["message"], phase=row["phase"], batch_id=row["id"])
                marker = (row["phase"], row.get("updated_at"), row["message"])
                if marker != previous_phase:
                    context.checkpoint(batch_id=row["id"], phase=row["phase"])
                    previous_phase = marker
        finally:
            iterator.close()
        return {"avatar_id": args["avatar_id"]}

    def regenerate(self, args, context):
        avatar = self.ready_avatar(args["avatar_id"])
        item = studio.gallery_item_by_id(str(self.root), avatar["id"], args["item_id"])
        if not item or item.get("origin") in {"legacy", "enhancement"} or not item.get("prompt"):
            raise ValueError("This image has no reusable generation recipe")
        if not context.saved.get("rejected"):
            item = studio.reject_gallery_item(str(self.root), avatar["id"], item["id"], args.get("reason", ""))
            context.checkpoint(rejected=True)
        prompt = context.saved.get("prompt")
        if not prompt:
            prompt = item["prompt"] + " Correct: " + (args.get("reason") or "identity and anatomy") if model_manager.DEV_MODE or item.get("origin") in {"console", "guided"} else avatar_gallery.repair_generation_prompt(item["prompt"], item["validation"])
            context.checkpoint(prompt=prompt)
        if context.saved.get("generated_path"):
            path, seed = context.saved["generated_path"], context.saved["seed"]
        else:
            context.emit("Regenerating selected image")
            extra = self.images([self.relative(path) for path in item.get("extra_references", []) if Path(path).is_file()])
            path, _, seed = self.gallery_image(avatar, {"path": item.get("reference_path")}, prompt,
                item.get("aspect", "864x1152 (3:4)"), item["id"], int(item.get("attempt", 1)) + 1, context,
                extras=extra, identity_only=item.get("origin") in {"console", "guided"})
            context.checkpoint(generated_path=path, seed=seed)
        context.emit("Checking regenerated image")
        with Image.open(path) as image:
            validation = self.validate(image.convert("RGB"), avatar, prompt)
        studio.update_gallery_item(str(self.root), avatar["id"], item["id"], prompt=prompt,
            generated_path=path, seed=seed, validation=validation, status="passed" if validation.get("pass") else "needs_review")
        return {"avatar_id": avatar["id"], "item_id": item["id"]}

    def enhance(self, args, context):
        avatar_id, source_id = args["avatar_id"], args["item_id"]
        item = studio.gallery_item_by_id(str(self.root), avatar_id, source_id)
        if not item:
            raise ValueError("The selected image no longer exists")
        item_id = f"enhance-{context.id}"
        if context.saved.get("image"):
            return {"avatar_id": avatar_id, "item_id": item_id, "image": context.saved["image"]}
        source = Image.open(item["generated_path"]).convert("RGB")
        seed = context.saved.get("seed") or secrets.randbelow(2**63)
        context.checkpoint(seed=seed)
        studio.record_gallery_item(str(self.root), avatar_id, {"id": item_id, "origin": "enhancement", "source_id": source_id,
            "source_path": item["generated_path"], "status": "enhancing", "attempt": 1, "prompt": avatar_enhance.PHOTO_PROMPT,
            "method": args["method"], "strength": args["strength"], "steps": args["steps"], "scale": args["scale"], "seed": str(seed)})
        def edit(tile, whole, prompt, seed, strength):
            context.check()
            engine = model_manager.ensure_model(model_manager.FLUX_MODEL_NAME, status_callback=context.emit)
            return engine.img2img([tile] if whole is None else [tile, whole], prompt, "", seed, 1.0, strength, args["steps"])
        try:
            for row in avatar_enhance.reconstruct(source, edit, args["scale"], args["strength"], seed,
                    lambda: context.store.get(context.id).get("cancel_requested", False), args["method"]):
                if row["image"] is None:
                    context.emit(f"FLUX pass {row['tile']}/{row['total']} / one output image", pass_index=row["tile"], total=row["total"])
                else:
                    path = studio.save_gallery_image(str(self.root), avatar_id, item_id, row["image"])
                    studio.update_gallery_item(str(self.root), avatar_id, item_id, generated_path=path,
                        status="no_change" if row["diagnostics"]["near_noop"] else "unchecked", diagnostics=row["diagnostics"],
                        processing_passes=row["total"], working_size=list(row["working_size"]), output_size=list(row["size"]))
                    context.checkpoint(image=self.relative(path))
        except avatar_enhance.EnhancementCancelled as exc:
            studio.update_gallery_item(str(self.root), avatar_id, item_id, status="stopped")
            raise Cancelled(str(exc)) from exc
        return {"avatar_id": avatar_id, "item_id": item_id, "image": context.saved.get("image")}
