"""Create repeatable synthetic fixtures for local browser QA, never cloud inference."""
from pathlib import Path
from PIL import Image, ImageDraw

from runtime_environment import provider
import avatar_studio as studio
import avatar_enhance
import workspace


def prepare(root):
    if provider() != "local":
        raise RuntimeError("Browser fixtures are local-only. Never run this against a cloud workspace.")
    root = Path(root).resolve()
    qa = root / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    source = qa / "supplied-preview.png"
    if not source.exists():
        image = Image.new("RGB", (768, 1024), (202, 222, 232))
        drawing = ImageDraw.Draw(image)
        for x in range(0, 768, 16):
            drawing.line((x, 0, x, 1024), fill=(130, 170, 180))
        drawing.ellipse((240, 100, 528, 388), fill=(185, 135, 105))
        drawing.rectangle((180, 400, 588, 900), fill=(40, 120, 90))
        image.save(source)
    avatar = next((row for row in studio.list_avatars(root) if row["name"] == "Enhancement QA"), None)
    if not avatar:
        avatar = studio.create_avatar(root, "Enhancement QA")
        with Image.open(source) as image:
            for part in ("face", "body"):
                studio.lock_reference(root, avatar["id"], part, image.convert("RGB"), dev_mode=True)
        avatar = studio.load_avatar(root, avatar["id"])
    items = studio.load_gallery_items(root, avatar["id"])
    if len(items) < 30:
        with Image.open(source) as opened:
            for index in range(30):
                identifier = f"qa-synthetic-{index}"
                if any(row["id"] == identifier for row in items):
                    continue
                path = studio.save_gallery_image(root, avatar["id"], identifier, opened.convert("RGB"))
                studio.record_gallery_item(root, avatar["id"], {"id": identifier, "origin": "console", "generated_path": path,
                    "prompt": f"Synthetic browser fixture {index}", "status": "unchecked", "seed": str(index)})
    if not any(row.get("origin") == "enhancement" for row in items):
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            path = studio.save_gallery_image(root, avatar["id"], "qa-enhancement", image.resize((1536, 2048)))
            studio.record_gallery_item(root, avatar["id"], {"id": "qa-enhancement", "origin": "enhancement", "generated_path": path,
                "source_path": str(source), "prompt": avatar_enhance.PHOTO_PROMPT, "status": "no_change", "seed": "0",
                "diagnostics": avatar_enhance.change_metrics(image, image)})
    return avatar["id"]


if __name__ == "__main__":
    print("Prepared local mock avatar:", prepare(workspace.get_save_dir()))
