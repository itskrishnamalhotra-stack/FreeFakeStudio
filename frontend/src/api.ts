export type Picture = {
  role?: "Outfit" | "Pose" | "Scene" | "Other person";
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
  source?: Picture | null;
  reconstruction_prompt?: string;
  diagnostics?: Record<string, unknown>;
};
export type ChatSummary = {
  id: string;
  avatar_id: string | null;
  title: string;
  created_at: number;
  updated_at: number;
  turn_count: number;
};
export type Conversation = ChatSummary & { turns: Turn[] };
export type Avatar = {
  identity_version?: string;
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
  batch_id?: string | null;
  regenerable: boolean;
  seed: string;
  diagnostics?: Record<string, unknown>;
  processing_passes?: number;
  manual_review?: unknown;
  validation?: unknown;
};
export type AvatarGender = "female" | "male" | "any";
export type GalleryBatch = {
  avatar_gender?: AvatarGender;
  id: string;
  name: string;
  theme: string;
  quantity: number;
  status: string;
  phase: string;
  archived?: boolean;
  created_at: string;
  completion_reason?: string;
  last_error?: string;
  message?: string;
  legacy_shared_references?: boolean;
  counts: Record<string, number>;
};
export type Resource = {
  id: string;
  name: string;
  image: Picture | null;
  kind: "identity" | "uploaded" | "search";
  state: "pending" | "approved" | "rejected";
  check_state: string;
  reason: string;
  used: boolean;
  committed: boolean;
  source_url?: string;
  validation?: unknown;
};
export type GalleryCounts = {
  generations: number;
  enhancements: number;
  resources: number;
  review: number;
};
export type GalleryPage = {
  items: GalleryItem[];
  total: number;
  counts: GalleryCounts;
};
export type ResourcePage = {
  items: Resource[];
  total: number;
  counts: Record<string, number>;
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
  app_settings: { gpu_memory_management: boolean };
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
  reconstruction_prompt: string;
  enhancement_profiles: { id: string; name: string; prompt: string }[];
  gallery_api: unknown;
};
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("ngrok-skip-browser-warning", "1");
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
