export type Picture = {
  path: string;
  url: string;
  thumb: string;
  name?: string;
  width?: number;
  height?: number;
};
export type Turn = {
  id: string;
  role: string;
  text: string;
  images: Picture[];
  seed?: string;
  recipe?: Record<string, unknown>;
};
export type Avatar = {
  id: string;
  name: string;
  face_locked: boolean;
  body_locked: boolean;
  face: Picture | null;
  body: Picture | null;
  face_candidate: Picture | null;
  body_candidate: Picture | null;
  face_prompt: string;
  body_prompt: string;
  face_specs: unknown;
  body_specs: unknown;
  chat: Turn[];
  creation_mode: string;
  guide: {
    active: boolean;
    stage: string;
    next_label: string;
    options: Record<string, string>;
    face_draft: Picture | null;
    body_draft: Picture | null;
  };
};
export type GalleryItem = {
  id: string;
  image: Picture;
  source: Picture | null;
  prompt: string;
  status: string;
  origin: string;
  regenerable: boolean;
  seed: string;
  diagnostics?: Record<string, unknown>;
  processing_passes?: number;
  validation?: unknown;
};
export type Job = {
  id: string;
  kind: string;
  status: string;
  message: string;
  error?: string;
  sync_error?: string;
  payload: Record<string, unknown>;
  result?: Record<string, unknown>;
  revision: number;
  cancel_requested?: boolean;
};
export type Config = {
  mock: boolean;
  provider: string;
  models: {
    name: string;
    capabilities: string[];
    defaults: {
      steps: number;
      cfg: number;
      denoise: number;
      max_steps: number;
    };
    status: string;
  }[];
  aspects: string[];
  modes: string[];
  mask_modes: string[];
  presets: string[];
  encoder_choices: string[];
  encoder: string;
  active_model: string | null;
  gallery_api: unknown;
};
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData))
    headers.set("Content-Type", "application/json");
  const response = await fetch("/api" + path, {
    ...init,
    headers,
    credentials: "same-origin",
  });
  const data = await response
    .json()
    .catch(() => ({ detail: "Server returned an unreadable response" }));
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail || data),
    );
  return data as T;
}
export const write = <T>(path: string, value: unknown = {}, method = "POST") =>
  api<T>(path, { method, body: JSON.stringify(value) });
export async function upload(file: File): Promise<Picture> {
  const body = new FormData();
  body.append("file", file);
  return api<Picture>("/uploads", { method: "POST", body });
}
export const active = (job: Job) => ["queued", "running"].includes(job.status);
export async function submit(value: Record<string, unknown>) {
  const body = JSON.stringify(value);
  let pending: { body: string; key: string } | null = null;
  try {
    pending = JSON.parse(
      sessionStorage.getItem("ffs-pending-request") || "null",
    );
  } catch {
    /* Discard a malformed browser cache. */
  }
  const key = pending?.body === body ? pending.key : crypto.randomUUID();
  sessionStorage.setItem("ffs-pending-request", JSON.stringify({ body, key }));
  const job = await api<Job>("/jobs", {
    method: "POST",
    headers: { "Idempotency-Key": key },
    body,
  });
  sessionStorage.removeItem("ffs-pending-request");
  return job;
}
