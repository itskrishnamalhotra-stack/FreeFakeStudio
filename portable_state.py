"""Non-destructive relocation of project-owned paths from older runtimes."""
import json
from pathlib import Path
import shutil


def relocate_results(root):
    root = Path(root).resolve()
    changed = []

    def convert(value):
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, str):
            normalized = value.replace("\\", "/")
            if "/results/" in normalized and "://" not in normalized:
                relative = normalized.split("/results/", 1)[1]
                candidate = (root / relative).resolve()
                if candidate.is_relative_to(root) and candidate.exists():
                    return str(candidate)
        return value

    for path in root.rglob("*.json"):
        if any(part in {"_thumbs", ".relocation-backup"} for part in path.parts):
            continue
        before = json.loads(path.read_text(encoding="utf-8"))
        after = convert(before)
        if before == after:
            continue
        backup = root / ".relocation-backup" / path.relative_to(root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
        temporary = path.with_suffix(".relocating")
        temporary.write_text(json.dumps(after, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
        changed.append(str(path.relative_to(root)))
    return changed
