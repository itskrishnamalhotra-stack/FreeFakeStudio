"""Private Google Drive mirror for notebook providers without Colab's mount.

Single-writer storage: never run two notebooks against the same folder at once.
Remote checksum checks reject overwrites when another session changed a file.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import threading
import time

import requests

API = "https://www.googleapis.com/drive/v3/files"
FOLDER = "application/vnd.google-apps.folder"
FIELDS = "id,name,mimeType,size,md5Checksum,modifiedTime"


def digest(path):
    value = hashlib.md5()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def safe_name(name):
    return bool(name and name not in {".", ".."} and not any(c in name for c in "/\\\x00"))


class DriveStore:
    def __init__(self, folder_id, root, client_id, client_secret, refresh_token, session=None, sleep=time.sleep):
        if not re.fullmatch(r"[A-Za-z0-9_-]{10,}", folder_id or ""):
            raise ValueError("Set GOOGLE_DRIVE_WORKSPACE_FOLDER_ID to the existing FreeFakeStudio folder ID")
        if not all((client_id, client_secret, refresh_token)):
            raise ValueError("Google Drive OAuth client ID, client secret and refresh token are required on Kaggle")
        self.folder_id, self.root = folder_id, Path(root).resolve()
        self.credentials = dict(client_id=client_id, client_secret=client_secret, refresh_token=refresh_token, grant_type="refresh_token")
        self.session, self.sleep = session or requests.Session(), sleep
        self.access, self.expires = "", 0
        self.lock = threading.RLock()
        self.index = {}
        self.scan_key, self.scan_time = None, 0
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / ".drive-sync.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    @classmethod
    def from_environment(cls, root=None):
        return cls(os.environ.get("GOOGLE_DRIVE_WORKSPACE_FOLDER_ID", ""), root or os.environ["FFS_WORKSPACE"],
                   os.environ.get("GOOGLE_DRIVE_CLIENT_ID", ""), os.environ.get("GOOGLE_DRIVE_CLIENT_SECRET", ""),
                   os.environ.get("GOOGLE_DRIVE_REFRESH_TOKEN", ""))

    def refresh(self):
        response = self.session.post("https://oauth2.googleapis.com/token", data=self.credentials, timeout=45)
        if response.status_code != 200:
            raise RuntimeError("Google Drive authorization failed. Renew the private OAuth refresh token; no saved work was deleted.")
        data = response.json()
        self.access, self.expires = data["access_token"], time.monotonic() + int(data.get("expires_in", 3600)) - 90

    def request(self, method, url, accepted=(200,), **kwargs):
        base_headers = kwargs.pop("headers", {})
        for attempt in range(6):
            if not self.access or time.monotonic() >= self.expires:
                self.refresh()
            headers = {**base_headers, "Authorization": "Bearer " + self.access}
            try:
                response = self.session.request(method, url, headers=headers, timeout=(30, 180), **kwargs)
            except requests.RequestException:
                if attempt == 5:
                    raise RuntimeError("Google Drive network unavailable. Local work remains saved; retry sync.") from None
                self.sleep(min(30, 2**attempt))
                continue
            if response.status_code in accepted:
                return response
            if response.status_code == 401 and attempt < 5:
                self.access = ""
            elif response.status_code in {429, 500, 502, 503, 504} and attempt < 5:
                self.sleep(min(30, 2**attempt))
            else:
                raise RuntimeError(f"Google Drive request failed (HTTP {response.status_code}). Check authorization, quota and folder access.")
        raise RuntimeError("Google Drive retry limit reached; local work is preserved")

    def children(self, parent):
        token = None
        while True:
            params = {"q": f"'{parent}' in parents and trashed = false", "fields": f"nextPageToken,files({FIELDS})", "pageSize": 1000}
            if token:
                params["pageToken"] = token
            data = self.request("GET", API, params=params).json()
            yield from data.get("files", [])
            token = data.get("nextPageToken")
            if not token:
                return

    def scan(self, select=lambda path, folder: True):
        found = {}
        def walk(parent, prefix=""):
            seen = set()
            for row in self.children(parent):
                name = row["name"]
                if not safe_name(name):
                    continue
                relative = prefix + name
                folder = row["mimeType"] == FOLDER
                if not select(relative, folder):
                    continue
                if name in seen:
                    raise RuntimeError(f"Duplicate names in Drive folder: {relative}. Resolve duplicates before starting.")
                seen.add(name)
                found[relative] = row
                if folder:
                    walk(row["id"], relative + "/")
        walk(self.folder_id)
        self.index = found
        return found

    def remember(self, relative, row, path):
        stat = path.stat()
        self.state[relative] = {"id": row["id"], "remote_md5": row.get("md5Checksum"), "local_md5": digest(path),
                                "mtime": stat.st_mtime_ns, "size": stat.st_size}
        self.save_state()

    def save_state(self):
        temporary = self.state_path.with_suffix(".part")
        temporary.write_text(json.dumps(self.state), encoding="utf-8")
        temporary.replace(self.state_path)

    def download(self, relative, row):
        path = self.root.joinpath(*PurePosixPath(relative).parts)
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Unsafe Drive path")
        prior = self.state.get(relative, {})
        if path.is_file() and prior.get("local_md5") and digest(path) != prior["local_md5"]:
            if row.get("md5Checksum") != prior.get("remote_md5"):
                raise RuntimeError(f"Drive conflict while restoring {relative}; both copies were preserved")
            print(f"[drive] Retaining unsynced local work: {relative}", flush=True)
            return
        if path.is_file() and path.stat().st_size == int(row.get("size", -1)) and digest(path) == row.get("md5Checksum"):
            self.remember(relative, row, path)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".drive-part")
        response = self.request("GET", API + "/" + row["id"], params={"alt": "media"}, stream=True)
        try:
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(4 * 1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        finally:
            response.close()
        if temporary.stat().st_size != int(row.get("size", -1)) or (row.get("md5Checksum") and digest(temporary) != row["md5Checksum"]):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Drive download verification failed for {relative}; original file was retained")
        temporary.replace(path)
        self.remember(relative, row, path)

    def restore(self, select):
        with self.lock:
            for relative, row in self.scan(select).items():
                if row["mimeType"] != FOLDER:
                    print(f"[drive] Restore {relative} ({int(row.get('size', 0)) / 1024**2:.1f} MB)", flush=True)
                    self.download(relative, row)

    def parent(self, relative):
        parent = self.folder_id
        prefix = ""
        for name in PurePosixPath(relative).parts[:-1]:
            prefix += name
            if prefix not in self.index:
                row = self.request("POST", API, accepted=(200, 201), json={"name": name, "mimeType": FOLDER, "parents": [parent]}, params={"fields": FIELDS}).json()
                self.index[prefix] = row
            parent = self.index[prefix]["id"]
            prefix += "/"
        return parent

    def upload(self, relative, path):
        prior = self.state.get(relative, {})
        remote = self.index.get(relative)
        checksum = digest(path)
        if remote and remote.get("md5Checksum") == checksum:
            self.remember(relative, remote, path)
            return
        if remote and remote.get("md5Checksum") != prior.get("remote_md5"):
            raise RuntimeError(f"Drive conflict: {relative} was changed by another session. Stop the other notebook; local work has not been overwritten.")
        if remote:
            remote = self.request("GET", API + "/" + remote["id"], params={"fields": FIELDS}).json()
            if remote.get("md5Checksum") != prior.get("remote_md5"):
                raise RuntimeError(f"Drive conflict: {relative} changed before upload. Local work was retained.")
        size = path.stat().st_size
        if not size:
            return
        # Pre-generated IDs make retrying an ambiguous create safe.
        file_id = remote["id"] if remote else prior.get("pending_id")
        if not file_id:
            file_id = self.request("GET", API + "/generateIds", params={"count": 1}).json()["ids"][0]
            self.state.setdefault(relative, {})["pending_id"] = file_id
            self.save_state()
        metadata = {} if remote else {"id": file_id, "name": path.name, "parents": [self.parent(relative)]}
        url = "https://www.googleapis.com/upload/drive/v3/files" + ("/" + file_id if remote else "")
        for restart in range(3):
            response = self.request("PATCH" if remote else "POST", url, accepted=(200, 201, 409),
                params={"uploadType": "resumable", "fields": FIELDS}, json=metadata,
                headers={"X-Upload-Content-Type": "application/octet-stream", "X-Upload-Content-Length": str(size)})
            if response.status_code == 409:
                row = self.request("GET", API + "/" + file_id, params={"fields": FIELDS}).json()
                if row.get("md5Checksum") == checksum:
                    self.index[relative] = row
                    self.remember(relative, row, path)
                    return
                raise RuntimeError("Drive upload ID conflict; saved local file was retained")
            location = response.headers["Location"]
            if not location.startswith("https://www.googleapis.com/"):
                raise RuntimeError("Unexpected Drive resumable upload host")
            offset = 0
            with path.open("rb") as stream:
                while offset < size:
                    stream.seek(offset)
                    chunk = stream.read(8 * 1024 * 1024)
                    try:
                        response = self.request("PUT", location, accepted=(200, 201, 308, 404), data=chunk,
                            headers={"Content-Length": str(len(chunk)), "Content-Range": f"bytes {offset}-{offset + len(chunk) - 1}/{size}"})
                    except RuntimeError:
                        response = self.request("PUT", location, accepted=(200, 201, 308, 404), data=b"",
                            headers={"Content-Length": "0", "Content-Range": f"bytes */{size}"})
                    if response.status_code == 404:
                        break
                    if response.status_code in {200, 201}:
                        row = response.json()
                        if row.get("md5Checksum") != checksum:
                            raise RuntimeError(f"Drive upload checksum mismatch for {relative}")
                        self.index[relative] = row
                        self.remember(relative, row, path)
                        return
                    match = re.fullmatch(r"bytes=0-(\d+)", response.headers.get("Range", ""))
                    following = int(match[1]) + 1 if match else 0
                    if following == offset:
                        raise RuntimeError("Drive upload made no progress; retry sync")
                    offset = following
        raise RuntimeError(f"Drive upload session expired repeatedly for {relative}; local work remains saved")

    def sync(self, directories=("results", "config", "manifests", "diagnostics")):
        with self.lock:
            if self.state_path.exists():
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            selected = set(directories)
            key = tuple(sorted(selected))
            if key != self.scan_key or time.monotonic() - self.scan_time >= 60:
                self.scan(lambda name, folder: any(name == prefix or name.startswith(prefix + "/") or (folder and prefix.startswith(name + "/")) for prefix in selected)
                          and not any(part in {"_thumbs", ".relocation-backup"} for part in PurePosixPath(name).parts))
                self.scan_key, self.scan_time = key, time.monotonic()
            files = [path for directory in directories for path in (self.root / directory).rglob("*")
                     if path.is_file() and not path.is_symlink() and not any(part.startswith(".") or part == "_thumbs" for part in path.relative_to(self.root).parts)
                     and not path.name.endswith((".part", ".tmp", ".pid"))]
            # Binary assets are committed before records that refer to them.
            for path in sorted(files, key=lambda value: (value.suffix == ".json", str(value))):
                relative = path.relative_to(self.root).as_posix()
                prior, stat = self.state.get(relative, {}), path.stat()
                remote = self.index.get(relative)
                if remote and prior.get("remote_md5") and remote.get("md5Checksum") != prior["remote_md5"]:
                    raise RuntimeError(f"Drive conflict: another session changed {relative}. Stop the other writer; local files are preserved.")
                if prior.get("mtime") == stat.st_mtime_ns and prior.get("size") == stat.st_size:
                    continue
                self.upload(relative, path)


def restore_selection(path, folder):
    parts = PurePosixPath(path).parts
    if any(part.startswith(".") or part in {"_thumbs", "__pycache__", "node_modules"} for part in parts):
        return False
    if parts[0] in {"results", "config", "manifests", "diagnostics"}:
        return not path.endswith((".pid", ".part", ".tmp"))
    if parts[0] == "bundles":
        return folder and (len(parts) == 1 or parts[1] == "caches") or len(parts) > 2 and parts[1] == "caches"
    if parts[0] == "ComfyUI":
        if folder:
            return len(parts) <= 2 and (len(parts) == 1 or parts[1] == "models") or len(parts) == 3 and parts[1] == "models"
        names = {"flux-2-klein-4b.safetensors", "qwen_3_4b_fp4_flux2.safetensors", "flux2-vae.safetensors"}
        return len(parts) >= 3 and parts[1] == "models" and (os.environ.get("FFS_MODEL_SET") == "all" or parts[-1] in names or "custom" in parts[-1].lower())
    return False
