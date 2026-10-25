"""CPU-only YuNet/SFace comparison. No implicit model downloads or GPU execution."""

import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image

_lock = threading.Lock()
_models = None


def install_models(folder):
    import hashlib
    from urllib.request import urlopen
    from runtime_environment import provider
    if provider() not in {"colab", "kaggle"}:
        raise RuntimeError("Identity model downloads require a cloud runtime; local tests use mocks.")
    revision = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
    files = (("face_detection_yunet", "face_detection_yunet_2023mar.onnx", "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
             ("face_recognition_sface", "face_recognition_sface_2021dec.onnx", "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for group, filename, digest in files:
        target = folder / filename
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            continue
        partial = target.with_suffix(".partial")
        try:
            with urlopen(f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{revision}/models/{group}/{filename}", timeout=120) as response, partial.open("wb") as output:
                count = 0
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > 40_000_000:
                        raise RuntimeError("Identity model download exceeds its expected size.")
                    output.write(chunk)
            if hashlib.sha256(partial.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Identity model checksum mismatch; existing model was preserved.")
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)
    os.environ["FFS_FACE_MODELS"] = str(folder)
    return str(folder)


def _load():
    global _models
    import cv2
    folder = Path(os.environ.get("FFS_FACE_MODELS", "models/avatar_identity"))
    detector = folder / "face_detection_yunet_2023mar.onnx"
    recognizer = folder / "face_recognition_sface_2021dec.onnx"
    if not detector.is_file() or not recognizer.is_file():
        raise RuntimeError("CPU identity models are missing. Run the updated Colab launcher.")
    if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
        raise RuntimeError("OpenCV face recognition is unavailable; repair the Colab OpenCV package.")
    if _models is None:
        _models = (cv2.FaceDetectorYN.create(str(detector), "", (640, 640), 0.85, 0.3, 5000, cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU),
                   cv2.FaceRecognizerSF.create(str(recognizer), "", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU))
    return _models


def features(image):
    import model_manager
    if model_manager.DEV_MODE:
        raise RuntimeError("Face inference is disabled in local mock mode.")
    with _lock:
        detector, recognizer = _load()
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                image = opened.convert("RGB")
        image = image.convert("RGB").copy()
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        frame = np.asarray(image)[:, :, ::-1].copy()
        detector.setInputSize((image.width, image.height))
        _, faces = detector.detect(frame)
        result = []
        for face in faces if faces is not None else []:
            if min(face[2], face[3]) < 48:
                continue
            crop = recognizer.alignCrop(frame, face)
            feature = recognizer.feature(crop).reshape(-1).astype(np.float32)
            norm = float(np.linalg.norm(feature))
            if np.isfinite(feature).all() and norm > 0:
                result.append({"embedding": feature / norm, "box": [float(x) for x in face[:4]],
                               "detector_score": float(face[-1])})
        return result


def detect_faces(image):
    """Bounding boxes only, using the already installed CPU detector, no embeddings."""
    import model_manager
    if model_manager.DEV_MODE:
        raise RuntimeError("Face inference is disabled in local mock mode.")
    with _lock:
        detector, _ = _load()
        image = image.convert("RGB").copy()
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        frame = np.asarray(image)[:, :, ::-1].copy()
        detector.setInputSize(image.size)
        _, faces = detector.detect(frame)
        return [{"box": [float(v) for v in face[:4]], "detector_score": float(face[-1]),
                 "coordinate_size": list(image.size)}
                for face in faces if min(face[2], face[3]) >= 24] if faces is not None else []


def calibrate(anchors, positives, negatives, extractor=features):
    reference = []
    for path in anchors:
        found = extractor(path)
        if len(found) != 1:
            raise ValueError("Each identity anchor must contain exactly one usable face.")
        reference.append(found[0]["embedding"])
    if not reference or len(positives) < 2 or len(negatives) < 2:
        raise ValueError("Use approved anchors, at least two same-person images and two different-person images.")
    scores = []
    for label, images in ((True, positives), (False, negatives)):
        for image in images:
            found = extractor(image)
            if len(found) != 1:
                raise ValueError("Each calibration image must contain exactly one usable face.")
            scores.append((label, max(float(np.dot(found[0]["embedding"], anchor)) for anchor in reference)))
    positive = [score for label, score in scores if label]
    negative = [score for label, score in scores if not label]
    held_out = len(positive) >= 4 and len(negative) >= 4
    train_positive = positive[:-2] if held_out else positive
    train_negative = negative[:-2] if held_out else negative
    lowest_match = min(train_positive)
    highest_other = max(train_negative)
    if lowest_match - highest_other < 0.08:
        raise ValueError("Calibration faces overlap or are too close. Review identities and add clearer, varied examples.")
    threshold = (lowest_match + highest_other) / 2
    verified = held_out and min(positive[-2:]) >= threshold + .04 and max(negative[-2:]) <= threshold - .04
    return {"threshold": threshold, "margin": 0.04,
            "positive_count": len(positives), "negative_count": len(negatives),
            "lowest_positive": lowest_match, "highest_negative": highest_other,
            "model": "opencv-sface-2021dec", "calibrated": bool(verified),
            "provisional": not verified, "validation_passed": bool(verified),
            "held_out_positive": positive[-2:] if held_out else [], "held_out_negative": negative[-2:] if held_out else [],
            "reason": "Independent hold-out samples passed" if verified else "Use at least four varied same-person and four different-person images; the last two in each group must pass independent validation"}


def compare(image, anchors, profile=None, extractor=features):
    try:
        references = []
        for path in anchors:
            faces = extractor(path)
            if len(faces) == 1:
                references.append(faces[0]["embedding"])
        faces = extractor(image)
        if not references or not faces:
            return {"status": "not_assessed", "reason": "No usable reference face or visible generated face"}
        scores = sorted((max(float(np.dot(face["embedding"], anchor)) for anchor in references) for face in faces), reverse=True)
        if not profile or not profile.get("calibrated") or not profile.get("validation_passed"):
            return {"status": "uncalibrated", "similarity": scores[0], "reason": "Calibrate this identity before automatic approval"}
        ambiguous = len(scores) > 1 and scores[0] - scores[1] < profile.get("margin", 0.04)
        passed = scores[0] >= profile["threshold"] and not ambiguous
        return {"status": "passed" if passed else "uncertain" if ambiguous else "failed", "similarity": scores[0],
                "threshold": profile["threshold"], "face_count": len(faces), "reason": "Multiple possible avatar faces" if ambiguous else "Calibrated cosine similarity; not an identity probability"}
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}
