"""Prepared-runtime persistence helpers for the Colab launcher.

This module intentionally uses only the Python standard library.  The launcher
can therefore inspect and repair its saved runtime before optional packages are
available in a fresh Colab VM.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path


SCHEMA_VERSION = 1
ARCHIVE_SUFFIX = ".tar.gz"


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(partial), str(path))


def load_json(path, default=None):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {} if default is None else default
    return value


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_text(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def detect_runtime_fingerprint():
    """Return the compatibility boundary for binary packages and GPU caches."""
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda or "none")
        cudnn_version = str(torch.backends.cudnn.version() or "none")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            capability = ".".join(map(str, torch.cuda.get_device_capability(0)))
        else:
            gpu_name = "none"
            capability = "none"
    except Exception:
        torch_version = "unavailable"
        cuda_version = "unavailable"
        cudnn_version = "unavailable"
        gpu_name = "unavailable"
        capability = "unavailable"

    driver = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
    ).splitlines()
    payload = {
        "schema": SCHEMA_VERSION,
        "python": platform.python_version(),
        "python_abi": getattr(sys.implementation, "cache_tag", "unknown"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch_version,
        "cuda": cuda_version,
        "cudnn": cudnn_version,
        "gpu": gpu_name,
        "gpu_capability": capability,
        "driver": driver[0].strip() if driver else "unavailable",
    }
    payload["id"] = hash_text(json.dumps(payload, sort_keys=True))[:20]
    return payload


def fingerprints_match(expected, current):
    if not isinstance(expected, dict) or not isinstance(current, dict):
        return False
    keys = (
        "schema",
        "python",
        "python_abi",
        "platform",
        "machine",
        "torch",
        "cuda",
        "cudnn",
        "gpu",
        "gpu_capability",
        "driver",
    )
    return all(expected.get(key) == current.get(key) for key in keys)


def git_revision(path):
    path = Path(path)
    if not (path / ".git").exists():
        return "missing"
    return _command_output(["git", "-C", str(path), "rev-parse", "HEAD"]) or "unknown"


def file_digest(path):
    path = Path(path)
    return sha256_file(path) if path.is_file() else "missing"


def source_signature(source, extra_paths=(), exclude_names=(), exclude_prefixes=()):
    source = Path(source)
    payload = {
        "revision": git_revision(source),
        "extras": {str(Path(item).name): file_digest(item) for item in extra_paths},
        "tree": tree_signature(source, exclude_names, exclude_prefixes),
    }
    return hash_text(json.dumps(payload, sort_keys=True))


def tree_signature(source, exclude_names=(), exclude_prefixes=()):
    source = Path(source)
    exclude_names = set(exclude_names)
    exclude_prefixes = tuple(str(item).strip("/") for item in exclude_prefixes)
    rows = []
    if source.exists():
        for root, dir_names, file_names in os.walk(source, followlinks=False):
            root_path = Path(root)
            relative_root = root_path.relative_to(source)
            dir_names[:] = [
                name for name in dir_names
                if not _excluded(relative_root / name, exclude_names, exclude_prefixes)
            ]
            for name in sorted(file_names):
                path = root_path / name
                if _excluded(path.relative_to(source), exclude_names, exclude_prefixes):
                    continue
                stat = path.stat()
                rows.append((path.relative_to(source).as_posix(), stat.st_size, stat.st_mtime_ns))
    return hash_text(json.dumps(rows, separators=(",", ":")))


def _excluded(relative, exclude_names, exclude_prefixes):
    parts = relative.parts
    if any(part in exclude_names for part in parts):
        return True
    posix = relative.as_posix()
    return any(posix == prefix or posix.startswith(prefix.rstrip("/") + "/") for prefix in exclude_prefixes)


def create_archive(source, archive_path, exclude_names=(), exclude_prefixes=()):
    """Create an atomic archive containing the children of ``source``."""
    source = Path(source).resolve()
    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial = archive_path.with_name(archive_path.name + ".part")
    partial.unlink(missing_ok=True)
    exclude_names = set(exclude_names)
    exclude_prefixes = tuple(str(item).replace("\\", "/").strip("/") for item in exclude_prefixes)

    with tarfile.open(partial, "w:gz", compresslevel=3, dereference=False) as archive:
        for root, dir_names, file_names in os.walk(source, followlinks=False):
            root_path = Path(root)
            relative_root = root_path.relative_to(source)
            dir_names[:] = [
                name
                for name in dir_names
                if not _excluded(relative_root / name, exclude_names, exclude_prefixes)
            ]
            for name in sorted(file_names):
                path = root_path / name
                relative = path.relative_to(source)
                if _excluded(relative, exclude_names, exclude_prefixes):
                    continue
                archive.add(path, arcname=relative.as_posix(), recursive=False)
    os.replace(str(partial), str(archive_path))
    return {
        "path": str(archive_path),
        "size": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
    }


def archive_is_valid(archive_path, expected_sha256=None):
    archive_path = Path(archive_path)
    if not archive_path.is_file() or archive_path.stat().st_size < 16:
        return False
    if expected_sha256 and sha256_file(archive_path) != expected_sha256:
        return False
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.getmembers()
    except (OSError, tarfile.TarError, EOFError):
        return False
    return True


def _safe_members(archive, destination):
    destination = Path(destination).resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if target != destination and destination not in target.parents:
            raise RuntimeError(f"Unsafe archive member: {member.name}")
        if member.issym() or member.islnk():
            link_target = (target.parent / member.linkname).resolve()
            if link_target != destination and destination not in link_target.parents:
                raise RuntimeError(f"Unsafe archive link: {member.name}")
        yield member


def restore_archive(archive_path, destination, expected_sha256=None, allowed_root=None):
    """Verify and atomically extract an archive into an ephemeral local root."""
    archive_path = Path(archive_path)
    destination = Path(destination)
    allowed_root = Path(allowed_root or destination.parent).resolve()
    resolved_destination = destination.resolve()
    if resolved_destination == allowed_root or allowed_root not in resolved_destination.parents:
        raise RuntimeError(f"Refusing to replace path outside local runtime: {destination}")
    if not archive_is_valid(archive_path, expected_sha256):
        raise RuntimeError(f"Prepared archive is missing or corrupt: {archive_path}")

    allowed_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=allowed_root))
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = list(_safe_members(archive, temporary))
            try:
                archive.extractall(temporary, members=members, filter="data")
            except TypeError:
                archive.extractall(temporary, members=members)
        previous = destination.with_name(destination.name + ".previous")
        if previous.exists():
            shutil.rmtree(previous)
        if destination.exists():
            os.replace(str(destination), str(previous))
        os.replace(str(temporary), str(destination))
        if previous.exists():
            shutil.rmtree(previous)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def ensure_source_bundle(
    component,
    source,
    destination,
    bundles_dir,
    manifests_dir,
    signature,
    force=False,
    exclude_names=(),
    exclude_prefixes=(),
):
    bundles_dir = Path(bundles_dir)
    manifests_dir = Path(manifests_dir)
    archive = bundles_dir / f"{component}{ARCHIVE_SUFFIX}"
    manifest_path = manifests_dir / f"{component}.json"
    manifest = load_json(manifest_path)
    reusable = (
        not force
        and manifest.get("schema") == SCHEMA_VERSION
        and manifest.get("signature") == signature
        and archive_is_valid(archive, manifest.get("archive_sha256"))
    )
    state = "restored"
    if not reusable:
        details = create_archive(source, archive, exclude_names, exclude_prefixes)
        manifest = {
            "schema": SCHEMA_VERSION,
            "component": component,
            "signature": signature,
            "archive": archive.name,
            "archive_size": details["size"],
            "archive_sha256": details["sha256"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_write_json(manifest_path, manifest)
        state = "rebuilt"
    try:
        restore_archive(
            archive,
            destination,
            manifest.get("archive_sha256"),
            allowed_root=Path(destination).parent,
        )
    except Exception:
        if state == "rebuilt":
            raise
        details = create_archive(source, archive, exclude_names, exclude_prefixes)
        manifest.update(
            archive_size=details["size"],
            archive_sha256=details["sha256"],
            repaired_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        atomic_write_json(manifest_path, manifest)
        restore_archive(archive, destination, details["sha256"], Path(destination).parent)
        state = "repaired"
    return state, manifest


def cache_environment(local_cache):
    local_cache = Path(local_cache)
    paths = {
        "HF_HOME": local_cache / "huggingface",
        "HF_HUB_CACHE": local_cache / "huggingface" / "hub",
        "HF_XET_CACHE": local_cache / "huggingface" / "xet",
        "HF_ASSETS_CACHE": local_cache / "huggingface" / "assets",
        "TORCH_HOME": local_cache / "torch",
        "TORCHINDUCTOR_CACHE_DIR": local_cache / "torchinductor",
        "TRITON_CACHE_DIR": local_cache / "triton",
        "TORCH_EXTENSIONS_DIR": local_cache / "torch_extensions",
        "CUDA_CACHE_PATH": local_cache / "cuda_compute",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    result = {name: str(path) for name, path in paths.items()}
    result.update(
        {
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
            "CUDA_CACHE_MAXSIZE": str(1024 * 1024 * 1024),
        }
    )
    return result


def restore_cache_bundle(cache_dir, archive, manifest_path, fingerprint, force_refresh=False):
    cache_dir = Path(cache_dir)
    archive = Path(archive)
    manifest = load_json(manifest_path)
    reusable = (
        not force_refresh
        and manifest.get("schema") == SCHEMA_VERSION
        and fingerprints_match(manifest.get("fingerprint"), fingerprint)
        and archive_is_valid(archive, manifest.get("archive_sha256"))
    )
    if reusable:
        try:
            restore_archive(archive, cache_dir, manifest["archive_sha256"], cache_dir.parent)
            return "restored"
        except (OSError, RuntimeError, tarfile.TarError):
            pass
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return "fresh"


def snapshot_cache_bundle(cache_dir, archive, manifest_path, fingerprint, force=False):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    exclusions = {".locks", "__pycache__"}
    content_signature = tree_signature(cache_dir, exclusions)
    previous = load_json(manifest_path)
    if (
        not force
        and previous.get("schema") == SCHEMA_VERSION
        and fingerprints_match(previous.get("fingerprint"), fingerprint)
        and previous.get("content_signature") == content_signature
        and archive_is_valid(archive, previous.get("archive_sha256"))
    ):
        return {**previous, "reused": True}
    details = create_archive(cache_dir, archive, exclude_names=exclusions)
    manifest = {
        "schema": SCHEMA_VERSION,
        "component": "runtime_caches",
        "fingerprint": fingerprint,
        "archive": Path(archive).name,
        "archive_size": details["size"],
        "archive_sha256": details["sha256"],
        "content_signature": content_signature,
        "reused": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def stage_file(source, destination, copy_to_ssd=True, reserve_bytes=4 * 1024**3):
    source = Path(source).resolve()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    copied = False
    if copy_to_ssd:
        usage = shutil.disk_usage(destination.parent)
        if usage.free >= source.stat().st_size + reserve_bytes:
            partial = destination.with_name(destination.name + ".part")
            partial.unlink(missing_ok=True)
            shutil.copyfile(source, partial)
            if partial.stat().st_size != source.stat().st_size:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"Incomplete SSD model copy: {source.name}")
            os.replace(str(partial), str(destination))
            copied = True
    if not copied:
        os.symlink(str(source), str(destination))
    return "copied" if copied else "linked"


def mirror_model_tree(
    drive_models,
    local_models,
    staged_sources=(),
    copy_to_ssd=True,
    required_sources=(),
):
    drive_models = Path(drive_models).resolve()
    local_models = Path(local_models)
    staged = {Path(path).resolve() for path in staged_sources}
    report = []
    report_by_source = {}

    def mirror(source):
        source = Path(source)
        relative = source.relative_to(drive_models)
        destination = local_models / relative
        mode = stage_file(source, destination, copy_to_ssd and source.resolve() in staged)
        item = {
            "source": str(source),
            "destination": str(destination),
            "size": source.stat().st_size,
            "mode": mode,
        }
        previous = report_by_source.get(str(source))
        if previous is None:
            report.append(item)
        else:
            previous.update(item)
            item = previous
        report_by_source[str(source)] = item
        return item

    for root, dir_names, file_names in os.walk(drive_models):
        root_path = Path(root)
        relative_root = root_path.relative_to(drive_models)
        (local_models / relative_root).mkdir(parents=True, exist_ok=True)
        dir_names[:] = [name for name in dir_names if name not in {".git", "__pycache__"}]
        for name in file_names:
            mirror(root_path / name)

    # Drive/FUSE directory listings can occasionally omit an entry even when a
    # direct lookup succeeds. Revisit every required checkpoint explicitly.
    for source in map(Path, required_sources):
        if not source.is_file():
            raise RuntimeError(f"Required persistent model file is unavailable: {source}")
        relative = source.relative_to(drive_models)
        destination = local_models / relative
        expected_size = source.stat().st_size
        if not destination.is_file() or destination.stat().st_size != expected_size:
            mirror(source)
        if not destination.is_file() or destination.stat().st_size != expected_size:
            raise RuntimeError(
                f"Required model mapping is incomplete: {source} -> {destination}"
            )
    return report


def _distribution_files(distribution_name):
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return []
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    platlib = Path(sysconfig.get_paths().get("platlib", purelib)).resolve()
    roots = {purelib, platlib}
    files = []
    for item in distribution.files or ():
        path = Path(distribution.locate_file(item)).resolve()
        for root in roots:
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if path.is_file() or path.is_symlink():
                files.append((path, relative))
            break
    return files


def distribution_closure(distributions, excluded=()):
    """Collect installed runtime dependencies without copying Colab's CUDA stack."""
    excluded_names = {
        re.sub(r"[-_.]+", "-", name).lower()
        for name in (
            "torch",
            "torchvision",
            "torchaudio",
            "triton",
            "numpy",
            "scipy",
            "pandas",
            "nvidia-cublas-cu12",
            "nvidia-cuda-runtime-cu12",
            "nvidia-cudnn-cu12",
            *excluded,
        )
    }
    queue = list(distributions)
    result = []
    seen = set()
    while queue:
        requested = queue.pop(0)
        normalized = re.sub(r"[-_.]+", "-", requested).lower()
        if normalized in seen or normalized in excluded_names:
            continue
        seen.add(normalized)
        try:
            distribution = importlib.metadata.distribution(requested)
        except importlib.metadata.PackageNotFoundError:
            continue
        canonical = distribution.metadata.get("Name") or requested
        result.append(canonical)
        for requirement in distribution.requires or ():
            if "extra ==" in requirement or "extra !=" in requirement:
                continue
            match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
            if match:
                dependency = match.group(1)
                dependency_normalized = re.sub(r"[-_.]+", "-", dependency).lower()
                if dependency_normalized not in seen and dependency_normalized not in excluded_names:
                    queue.append(dependency)
    return result


def build_python_overlay(distributions, overlay_dir):
    overlay_dir = Path(overlay_dir)
    temporary = overlay_dir.with_name(overlay_dir.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    copied = 0
    included = []
    for distribution_name in distribution_closure(distributions):
        files = _distribution_files(distribution_name)
        if not files:
            continue
        included.append(distribution_name)
        for source, relative in files:
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                shutil.copy2(source.resolve(), destination)
            else:
                shutil.copy2(source, destination)
            copied += 1
    if overlay_dir.exists():
        shutil.rmtree(overlay_dir)
    os.replace(str(temporary), str(overlay_dir))
    return {"distributions": included, "files": copied}


def restore_environment_bundle(overlay_dir, archive, manifest_path, fingerprint, requirements_digest):
    manifest = load_json(manifest_path)
    reusable = (
        manifest.get("schema") == SCHEMA_VERSION
        and manifest.get("requirements_digest") == requirements_digest
        and fingerprints_match(manifest.get("fingerprint"), fingerprint)
        and archive_is_valid(archive, manifest.get("archive_sha256"))
    )
    if not reusable:
        overlay_dir = Path(overlay_dir)
        if overlay_dir.exists():
            shutil.rmtree(overlay_dir)
        return False
    try:
        restore_archive(archive, overlay_dir, manifest["archive_sha256"], Path(overlay_dir).parent)
    except (OSError, RuntimeError, tarfile.TarError):
        overlay_dir = Path(overlay_dir)
        if overlay_dir.exists():
            shutil.rmtree(overlay_dir)
        return False
    if str(overlay_dir) not in sys.path:
        sys.path.insert(0, str(overlay_dir))
    return True


def snapshot_environment_bundle(
    distributions,
    overlay_dir,
    archive,
    manifest_path,
    fingerprint,
    requirements_digest,
):
    overlay_report = build_python_overlay(distributions, overlay_dir)
    details = create_archive(overlay_dir, archive, exclude_names={"__pycache__"})
    manifest = {
        "schema": SCHEMA_VERSION,
        "component": "python_overlay",
        "fingerprint": fingerprint,
        "requirements_digest": requirements_digest,
        "archive": Path(archive).name,
        "archive_size": details["size"],
        "archive_sha256": details["sha256"],
        "distributions": overlay_report["distributions"],
        "file_count": overlay_report["files"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def validate_wheelhouse(path):
    path = Path(path)
    removed = []
    if not path.exists():
        return removed
    for wheel in path.glob("*.whl"):
        try:
            with zipfile.ZipFile(wheel) as archive:
                if archive.testzip() is not None:
                    raise zipfile.BadZipFile("CRC failure")
        except (OSError, zipfile.BadZipFile):
            wheel.unlink(missing_ok=True)
            removed.append(wheel.name)
    return removed


def readiness_payload(path):
    value = load_json(path)
    return value if isinstance(value, dict) else {}


def write_readiness(path, **values):
    payload = {
        "schema": SCHEMA_VERSION,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **values,
    }
    atomic_write_json(path, payload)
    return payload
