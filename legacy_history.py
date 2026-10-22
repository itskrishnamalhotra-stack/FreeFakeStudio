"""Convert saved Gradio chat markup into safe structured turns, retaining originals."""
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote


class TurnParser(HTMLParser):
    def __init__(self, root):
        super().__init__(convert_charrefs=True)
        self.root, self.role, self.text, self.images = root, "assistant", [], []
        self.depth, self.capture = 0, None

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        classes = attrs.get("class", "").split()
        if "ffs-turn-user" in classes:
            self.role = "user"
        if "ffs-bubble" in classes:
            self.capture = self.depth
        if tag in {"a", "img"}:
            value = unquote(attrs.get("href" if tag == "a" else "src", ""))
            if "/file=" in value:
                value = value.split("/file=", 1)[1].split("?", 1)[0]
                suffix = value.replace("\\", "/").partition("/results/")[2]
                candidate = self.root / suffix if suffix else Path(value)
                if candidate.resolve().is_relative_to(self.root) and candidate.is_file():
                    relative = candidate.resolve().relative_to(self.root).as_posix()
                    if relative not in self.images:
                        self.images.append(relative)
        if tag == "br" and self.capture is not None:
            self.text.append("\n")
        if tag not in {"img", "br", "input", "hr", "meta", "link"}:
            self.depth += 1

    def handle_endtag(self, tag):
        self.depth = max(0, self.depth - 1)
        if self.capture is not None and self.depth <= self.capture:
            self.capture = None

    def handle_data(self, data):
        if self.capture is not None:
            self.text.append(data)


def migrate(root):
    root = Path(root).resolve()
    path = root / "_sessions" / "current_chat.json"
    if not path.exists():
        return {"version": 1, "turns": []}
    original = json.loads(path.read_text(encoding="utf-8"))
    turns = []
    for index, markup in enumerate(original.get("history", [])):
        if not isinstance(markup, str):
            continue
        parser = TurnParser(root)
        parser.feed(markup)
        turns.append({"id": f"imported-{index}", "role": parser.role, "text": "".join(parser.text).strip(), "images": parser.images})
    return {"version": 1, "turns": turns, "imported_from": "current_chat.json"}
