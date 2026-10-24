"""Resumable identity regression collection, human labels and honest metrics."""

import time
import uuid

import avatar_identity as identity
import avatar_studio as studio

CASES = (
    "Front-facing daylight portrait, neutral expression", "Three-quarter portrait, natural smile",
    "Profile portrait in soft daylight", "Portrait in warm indoor evening lighting",
    "Portrait outdoors in cool overcast light", "Full-body front view in a plain cotton outfit",
    "Full-body side view, walking naturally", "Seated full-body image with both hands visible",
    "Full-body image with a different hairstyle and unchanged facial identity",
    "Wide environmental portrait with a small but visible face",
    "The avatar standing beside a clearly different adult person; keep identities distinct",
    "Rear view of the avatar looking toward the horizon, face not visible",
)


def load(root, avatar_id):
    return identity._read(studio.avatar_path(root, avatar_id) / "benchmark.json")


def _save(root, avatar_id, data):
    studio._atomic_json(studio.avatar_path(root, avatar_id) / "benchmark.json", data)
    return data


def stop(root, avatar_id):
    data = load(root, avatar_id)
    if data:
        data["stop_requested"] = True
        _save(root, avatar_id, data)


def run(root, avatar_id, generate, validate, dev=False, check=None, run_id=None):
    avatar = identity.freeze(root, studio.load_avatar(root, avatar_id))
    if not avatar.get("identity_version"):
        raise ValueError("Confirm face and body before running the benchmark.")
    data = load(root, avatar_id)
    if not data or (data["status"] == "complete" and data.get("job_id") != run_id):
        if data:
            studio._atomic_json(studio.avatar_path(root, avatar_id) / "benchmarks" / (data["id"] + ".json"), data)
        data = {"id": uuid.uuid4().hex[:12], "job_id": run_id, "identity_version": avatar["identity_version"], "status": "running",
                "dev_mode": dev, "cases": [{"scene": scene, "seed": 1000 + index, "status": "pending"} for index, scene in enumerate(CASES)]}
    avatar = identity.pin(root, avatar, data["identity_version"])
    import avatar_quality
    if "quality_profile" not in data:
        data["quality_profile"] = avatar_quality.profile(root, avatar)
    avatar = {**avatar, "_quality_profile": data["quality_profile"]}
    data.update(status="running", stop_requested=False)
    _save(root, avatar_id, data)
    try:
        for index, case in enumerate(data["cases"]):
            if check:
                check()
            if load(root, avatar_id).get("stop_requested"):
                data["status"] = "stopped"
                break
            if case["status"] in {"passed", "needs_review"}:
                continue
            started = time.monotonic()
            case["status"] = "running"
            yield _save(root, avatar_id, data)
            if not case.get("path"):
                image = generate(avatar, case["scene"], case["seed"])
                case["path"] = studio.save_gallery_image(root, avatar_id, f"benchmark-{data['id']}-{index}", image)
                _save(root, avatar_id, data)
            from PIL import Image
            if check:
                check()
            with Image.open(case["path"]) as image:
                verdict = validate(image.convert("RGB"), avatar, case["scene"])
            case.update(validation=verdict, status="passed" if verdict.get("pass") is True else "needs_review",
                        seconds=round(time.monotonic() - started, 3), recipe=identity.recipe(avatar, identity.select_references(avatar, case["scene"])))
            studio.record_gallery_item(root, avatar_id, {"id": f"benchmark-{data['id']}-{index}",
                "origin": "benchmark", "prompt": case["scene"], "generated_path": case["path"],
                "status": case["status"], "validation": verdict, "seed": case["seed"],
                "identity_version": avatar["identity_version"], "recipe": case["recipe"]})
            yield _save(root, avatar_id, data)
        else:
            data["status"] = "complete"
        yield _save(root, avatar_id, data)
    except Exception as exc:
        data.update(status="paused", error=str(exc))
        _save(root, avatar_id, data)
        raise
    finally:
        if data["status"] == "running":
            data["status"] = "stopped"
            _save(root, avatar_id, data)


def label(root, avatar_id, index, correct):
    if correct not in {"correct", "wrong", "uncertain"}:
        raise ValueError("Choose a human identity assessment.")
    data = load(root, avatar_id)
    if not data or not 0 <= int(index) < len(data["cases"]):
        raise ValueError("Benchmark case not found.")
    data["cases"][int(index)]["human"] = correct
    return _save(root, avatar_id, data)


def report(data):
    if not data:
        return {"status": "not_run"}
    cases = data["cases"]
    reviewed = [case for case in cases if case.get("human") in {"correct", "wrong"}]
    return {"status": data["status"], "mock_not_quality_evidence": data["dev_mode"],
            "completed": sum(case["status"] in {"passed", "needs_review"} for case in cases),
            "automatic_passes": sum(case["status"] == "passed" for case in cases),
            "human_reviewed": len(reviewed), "false_acceptances": sum(case["status"] == "passed" and case["human"] == "wrong" for case in reviewed),
            "false_rejections": sum(case["status"] != "passed" and case["human"] == "correct" for case in reviewed),
            "total_seconds": sum(case.get("seconds", 0) for case in cases), "identity_version": data["identity_version"]}
