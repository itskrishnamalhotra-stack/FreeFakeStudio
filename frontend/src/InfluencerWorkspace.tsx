import React, { useEffect, useRef, useState } from "react";
import {
  Check,
  Download,
  Loader2,
  Plus,
  RefreshCw,
  Save,
  ScanFace,
  Send,
  Trash2,
  Upload,
  WandSparkles,
  X,
} from "lucide-react";
import { api, write, upload, Avatar, GalleryItem, Job, Picture } from "./api";

type Policy = "required" | "important" | "flexible";
type Candidate = {
  id: string;
  view: string;
  approved: boolean;
  image: Picture;
};
type Case = {
  index: number;
  scene: string;
  seed: number;
  status: string;
  human?: string;
  image: Picture | null;
  validation?: unknown;
};
type Identity = {
  version: string;
  views: string[];
  versions: { version: string; created_at: string }[];
  candidates: Candidate[];
  attribute_keys: string[];
  profile: {
    policies?: Record<string, Policy>;
    face?: {
      calibrated?: boolean;
      threshold?: number;
      mock_not_quality_evidence?: boolean;
      provisional?: boolean;
      reason?: string;
    };
  };
  calibration: { positive: Picture[]; negative: Picture[] };
  benchmark: Record<string, string | number | boolean>;
  cases: Case[];
};
type Post = {
  id: string;
  image_id: string;
  caption: string;
  planned_at: string;
  public_url: string;
  status: string;
  image: Picture;
  media_id?: string;
  permalink?: string;
};
type Content = {
  persona: Record<string, string>;
  posts: Post[];
  instagram_configured: boolean;
  scheduling_enabled: boolean;
};
type Props = {
  avatar: Avatar;
  tab: "identity" | "quality" | "content";
  busy: boolean;
  jobs: Job[];
  action: (value: Record<string, unknown>) => Promise<Job | undefined>;
  onError: (error: unknown) => void;
  view: (pictures: Picture[], index?: number) => void;
  onSaved: () => Promise<void>;
};
function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      {React.isValidElement(children)
        ? React.cloneElement(
            children as React.ReactElement<{ "aria-label": string }>,
            { "aria-label": label },
          )
        : children}
    </label>
  );
}
function ImageFiles({
  label,
  pictures,
  onChange,
  view,
  disabled,
}: {
  label: string;
  pictures: Picture[];
  onChange: (images: Picture[]) => void;
  view: Props["view"];
  disabled: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [loading, setLoading] = useState(false),
    [error, setError] = useState("");
  async function add(files: FileList | null) {
    if (!files) return;
    setLoading(true);
    setError("");
    try {
      const added: Picture[] = [];
      for (const file of Array.from(files).slice(0, 12 - pictures.length))
        added.push(await upload(file));
      onChange([...pictures, ...added]);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
      if (input.current) input.current.value = "";
    }
  }
  return (
    <div className="calibration-group">
      <div className="section-heading">
        <h3>{label}</h3>
        <span className="muted">{pictures.length}/12</span>
        <button
          disabled={disabled || loading || pictures.length >= 12}
          onClick={() => input.current?.click()}
        >
          <Upload size={15} />
          Add images
        </button>
      </div>
      <input
        ref={input}
        className="hidden"
        type="file"
        accept="image/png,image/jpeg,image/webp"
        multiple
        aria-label={label}
        onChange={(e) => add(e.target.files)}
      />
      {error && <p role="alert">{error}</p>}
      <div className="calibration-images">
        {pictures.map((pic, index) => (
          <figure key={pic.path}>
            <button onClick={() => view(pictures, index)}>
              <img src={pic.thumb} alt={label + " " + (index + 1)} />
            </button>
            <button
              className="icon-button"
              title={"Remove " + label + " " + (index + 1)}
              aria-label={"Remove " + label + " " + (index + 1)}
              disabled={disabled}
              onClick={() => onChange(pictures.filter((_, i) => i !== index))}
            >
              <X size={14} />
            </button>
          </figure>
        ))}
      </div>
    </div>
  );
}
export function InfluencerWorkspace({
  avatar,
  tab,
  busy,
  jobs,
  action,
  onError,
  view,
  onSaved,
}: Props) {
  const base = "/avatars/" + avatar.id;
  const [identity, setIdentity] = useState<Identity | null>(null),
    [content, setContent] = useState<Content | null>(null);
  const [images, setImages] = useState<GalleryItem[]>([]),
    [imageTotal, setImageTotal] = useState(0);
  const [loading, setLoading] = useState(false),
    [saving, setSaving] = useState(false),
    [error, setError] = useState("");
  const [candidateView, setCandidateView] = useState("front"),
    [details, setDetails] = useState(""),
    [version, setVersion] = useState("");
  const [policies, setPolicies] = useState<Record<string, Policy>>({});
  const [positive, setPositive] = useState<Picture[]>([]),
    [negative, setNegative] = useState<Picture[]>([]);
  const [persona, setPersona] = useState<Record<string, string>>({}),
    [brief, setBrief] = useState(""),
    [caption, setCaption] = useState("");
  const [imageId, setImageId] = useState(""),
    [planned, setPlanned] = useState(""),
    [publicUrl, setPublicUrl] = useState("");
  const [editing, setEditing] = useState<string | null>(null),
    [useApi, setUseApi] = useState(false),
    [status, setStatus] = useState("");
  const [captionJob, setCaptionJob] = useState<string | null>(null);
  const [captionStyle, setCaptionStyle] = useState("auto");
  const candidateInput = useRef<HTMLInputElement>(null);
  const calibrationLoaded = useRef(false);
  const policyDirty = useRef(false),
    personaDirty = useRef(false);
  const ready = avatar.face_locked && avatar.body_locked;
  const completed = jobs
    .filter(
      (j) =>
        j.payload.avatar_id === avatar.id &&
        !["queued", "running"].includes(j.status),
    )
    .map((j) => j.id + j.status)
    .join(":");
  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    setLoading(true);
    setError("");
    Promise.all([
      api<Identity>(base + "/identity"),
      api<Content>(base + "/content"),
      api<{ items: GalleryItem[]; total: number }>(base + "/gallery?limit=100"),
    ])
      .then(([id, data, gallery]) => {
        if (cancelled) return;
        setIdentity(id);
        if (!calibrationLoaded.current) {
          setPositive(id.calibration.positive.filter(Boolean));
          setNegative(id.calibration.negative.filter(Boolean));
          calibrationLoaded.current = true;
        }
        setVersion(id.version);
        if (!policyDirty.current) {
          setPolicies(id.profile.policies || {});
        }
        setContent(data);
        if (!personaDirty.current) setPersona(data.persona);
        setImages((old) => [
          ...gallery.items,
          ...old.filter(
            (image) =>
              image.id === imageId &&
              !gallery.items.some((row) => row.id === image.id),
          ),
        ]);
        setImageTotal(gallery.total);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [base, ready, completed]);
  useEffect(() => {
    const job = jobs.find((j) => j.id === captionJob);
    if (job?.status === "completed" && job.result?.caption) {
      setCaption(String(job.result.caption));
      setCaptionJob(null);
    }
  }, [jobs, captionJob]);
  async function mutate(path: string, data: unknown, method = "POST") {
    setSaving(true);
    setError("");
    setStatus("");
    try {
      await write(base + path, data, method);
      const [id, saved] = await Promise.all([
        api<Identity>(base + "/identity"),
        api<Content>(base + "/content"),
      ]);
      setIdentity(id);
      setVersion(id.version);
      setContent(saved);
      if (path === "/quality") policyDirty.current = false;
      if (path === "/persona") personaDirty.current = false;
      setStatus("Saved");
      await onSaved();
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      setSaving(false);
    }
  }
  async function run(kind: string, args: Record<string, unknown> = {}) {
    setStatus("");
    return action({ kind, avatar_id: avatar.id, ...args });
  }
  async function addCandidate(files: FileList | null) {
    if (!files?.[0]) return;
    setSaving(true);
    try {
      const pic = await upload(files[0]);
      await mutate("/identity", {
        action: "upload",
        image: pic.path,
        view: candidateView,
      });
    } catch (e) {
      onError(e);
    } finally {
      setSaving(false);
      if (candidateInput.current) candidateInput.current.value = "";
    }
  }
  async function moreImages() {
    try {
      const data = await api<{ items: GalleryItem[]; total: number }>(
        base + "/gallery?limit=100&offset=" + images.length,
      );
      setImages((old) => [...old, ...data.items]);
      setImageTotal(data.total);
    } catch (e) {
      onError(e);
    }
  }
  const disabled = busy || saving || loading;
  if (!ready)
    return (
      <div className="influencer-workspace">
        <p role="status">Confirm the face and body first.</p>
      </div>
    );
  if (!identity || !content)
    return (
      <div className="influencer-workspace">
        {error ? <p role="alert">{error}</p> : <Loader2 className="spin" />}
      </div>
    );
  return (
    <div className="influencer-workspace" aria-busy={loading}>
      {error && (
        <p className="alert" role="alert">
          {error}
        </p>
      )}
      {status && (
        <p className="muted" role="status">
          {status}
        </p>
      )}
      {tab === "identity" && (
        <>
          <section className="identity-version-bar">
            <div>
              <h2>Identity references</h2>
              <span className="muted">Version {identity.version}</span>
            </div>
            <Field label="Saved version">
              <select
                value={version}
                onChange={(e) => setVersion(e.target.value)}
              >
                {identity.versions.map((v) => (
                  <option key={v.version} value={v.version}>
                    {v.created_at} · {v.version.slice(0, 8)}
                  </option>
                ))}
              </select>
            </Field>
            <button
              disabled={disabled || version === identity.version}
              onClick={() =>
                window.confirm(
                  "Restore this identity version and its approved references?",
                ) && mutate("/identity", { action: "activate", version })
              }
            >
              <RefreshCw size={15} />
              Restore version
            </button>
            <a className="button" href={"/api" + base + "/identity/export"}>
              <Download size={15} />
              Export pack
            </a>
          </section>
          <section className="identity-create">
            <div className="influencer-fields">
              <Field label="Reference view">
                <select
                  value={candidateView}
                  onChange={(e) => setCandidateView(e.target.value)}
                >
                  {identity.views.map((v) => (
                    <option key={v} value={v}>
                      {v.replaceAll("-", " ")}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Additional details">
                <input
                  value={details}
                  onChange={(e) => setDetails(e.target.value)}
                  maxLength={12000}
                />
              </Field>
              <button
                className="primary"
                disabled={disabled}
                onClick={() =>
                  run("identity_view", { view: candidateView, details })
                }
              >
                <WandSparkles size={16} />
                Generate reference
              </button>
              <button
                disabled={disabled}
                onClick={() => candidateInput.current?.click()}
              >
                <Upload size={16} />
                Upload reference
              </button>
              <input
                ref={candidateInput}
                aria-label="Upload identity reference"
                className="hidden"
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={(e) => addCandidate(e.target.files)}
              />
            </div>
          </section>
          <section>
            <div className="section-heading">
              <h3>Approved anchors</h3>
              <span className="muted">Face + body</span>
            </div>
            <div className="identity-anchor-grid">
              {[avatar.face, avatar.body].map(
                (pic, index) =>
                  pic && (
                    <button key={pic.path} onClick={() => view([pic])}>
                      <img
                        src={pic.thumb}
                        alt={index ? "Confirmed body" : "Confirmed face"}
                      />
                      <span>
                        {index ? "Body" : "Face"}
                        <Check size={14} />
                      </span>
                    </button>
                  ),
              )}
            </div>
          </section>
          <section>
            <div className="section-heading">
              <h3>Reference pack</h3>
              <span className="muted">
                {identity.candidates.filter((c) => c.approved).length} approved
                / {identity.candidates.length} candidates
              </span>
            </div>
            <div className="identity-candidates">
              {identity.candidates.map((candidate, index) => (
                <article key={candidate.id}>
                  <button
                    className="identity-candidate-image"
                    onClick={() =>
                      view(
                        identity.candidates.map((c) => c.image),
                        index,
                      )
                    }
                  >
                    <img
                      src={candidate.image.thumb}
                      alt={candidate.view + " candidate"}
                    />
                  </button>
                  <div className="section-heading">
                    <strong>{candidate.view.replaceAll("-", " ")}</strong>
                    <span className="status">
                      {candidate.approved ? "Approved" : "Needs review"}
                    </span>
                  </div>
                  <div className="button-row">
                    <button
                      disabled={disabled}
                      aria-pressed={candidate.approved}
                      onClick={() =>
                        mutate("/identity", {
                          action: "approve",
                          candidate_id: candidate.id,
                          approved: !candidate.approved,
                        })
                      }
                    >
                      <Check size={15} />
                      {candidate.approved ? "Unapprove" : "Approve identity"}
                    </button>
                    <button
                      disabled={disabled}
                      title="Remove reference"
                      aria-label="Remove reference"
                      onClick={() =>
                        window.confirm(
                          "Remove this candidate? Saved identity versions remain available.",
                        ) &&
                        mutate("/identity", {
                          action: "remove",
                          candidate_id: candidate.id,
                        })
                      }
                    >
                      <Trash2 size={15} />
                    </button>
                  </div>
                </article>
              ))}
            </div>
            {!identity.candidates.length && (
              <p className="muted">No additional references.</p>
            )}
          </section>
        </>
      )}
      {tab === "quality" && (
        <>
          <section>
            <div className="section-heading">
              <h2>Identity calibration</h2>
              <span className="status">
                {identity.profile.face?.calibrated
                  ? "Calibrated"
                  : identity.profile.face?.provisional
                    ? "Provisional"
                    : "Not calibrated"}
              </span>
              {identity.profile.face?.mock_not_quality_evidence && (
                <span className="status">Mock result</span>
              )}
            </div>
            <div className="calibration-columns">
              <ImageFiles
                label="Same person"
                pictures={positive}
                onChange={setPositive}
                view={view}
                disabled={disabled}
              />
              <ImageFiles
                label="Different people"
                pictures={negative}
                onChange={setNegative}
                view={view}
                disabled={disabled}
              />
            </div>
            <button
              disabled={disabled || positive.length < 2 || negative.length < 2}
              onClick={() =>
                run("calibrate", {
                  positives: positive.map((p) => p.path),
                  negatives: negative.map((p) => p.path),
                })
              }
            >
              <ScanFace size={16} />
              Calibrate identity
            </button>
            {identity.profile.face?.threshold !== undefined && (
              <p className="muted">
                Cosine threshold: {identity.profile.face.threshold.toFixed(3)}
              </p>
            )}
            {identity.profile.face?.reason && (
              <p className="muted" role="status">
                {identity.profile.face.reason}
              </p>
            )}
            {identity.profile.face && (
              <details>
                <summary>Calibration evidence</summary>
                <pre>{JSON.stringify(identity.profile.face, null, 2)}</pre>
              </details>
            )}
          </section>
          <section>
            <div className="section-heading">
              <h2>Consistency policy</h2>
              <button
                disabled={disabled}
                onClick={() => mutate("/quality", { policies }, "PATCH")}
              >
                <Save size={15} />
                Save policy
              </button>
            </div>
            <div className="policy-grid">
              {identity.attribute_keys.map((key) => (
                <Field key={key} label={key.replaceAll("_", " ")}>
                  <select
                    value={policies[key] || "flexible"}
                    onChange={(e) => {
                      policyDirty.current = true;
                      setPolicies((old) => ({
                        ...old,
                        [key]: e.target.value as Policy,
                      }));
                    }}
                  >
                    <option value="required">Required match</option>
                    <option value="important">Important match</option>
                    <option value="flexible">Flexible</option>
                  </select>
                </Field>
              ))}
            </div>
          </section>
          <section>
            <div className="section-heading">
              <h2>Consistency test</h2>
              <span className="status">
                {String(identity.benchmark.status).replaceAll("_", " ")}
              </span>
              <button
                className="primary"
                disabled={disabled}
                onClick={() => run("benchmark")}
              >
                <RefreshCw size={15} />
                {identity.benchmark.status === "complete" ||
                identity.benchmark.status === "not_run"
                  ? "Run 12-image test"
                  : "Resume test"}
              </button>
            </div>
            <div className="benchmark-metrics">
              {[
                "completed",
                "automatic_passes",
                "human_reviewed",
                "false_acceptances",
                "false_rejections",
              ].map((key) => (
                <div key={key}>
                  <strong>{String(identity.benchmark[key] ?? 0)}</strong>
                  <span>{key.replaceAll("_", " ")}</span>
                </div>
              ))}
            </div>
            {identity.benchmark.mock_not_quality_evidence && (
              <p className="muted">Mock results, not image-quality evidence.</p>
            )}
            <div className="benchmark-cases">
              {identity.cases.map((row) => (
                <article key={row.index}>
                  {row.image && (
                    <button onClick={() => view([row.image!])}>
                      <img src={row.image.thumb} alt={row.scene} />
                    </button>
                  )}
                  <div>
                    <strong>{row.scene}</strong>
                    <p className="muted">
                      {row.status.replaceAll("_", " ")} · Seed {row.seed}
                    </p>
                    <Field label={"Human assessment " + (row.index + 1)}>
                      <select
                        disabled={disabled || !row.image}
                        value={row.human || "uncertain"}
                        onChange={(e) =>
                          mutate("/identity", {
                            action: "label",
                            index: row.index,
                            label: e.target.value,
                          })
                        }
                      >
                        <option value="uncertain">
                          Uncertain / not reviewed
                        </option>
                        <option value="correct">Correct identity</option>
                        <option value="wrong">Wrong identity</option>
                      </select>
                    </Field>
                    {!!row.validation && (
                      <details>
                        <summary>Check results</summary>
                        <pre>{JSON.stringify(row.validation, null, 2)}</pre>
                      </details>
                    )}
                  </div>
                </article>
              ))}
            </div>
          </section>
        </>
      )}
      {tab === "content" && (
        <>
          <section>
            <details open={!content.persona.niche}>
              <summary>Persona & voice</summary>
              <div className="persona-form">
                {["niche", "tone", "values", "disclosure"].map((key) => (
                  <Field key={key} label={key[0].toUpperCase() + key.slice(1)}>
                    <input
                      maxLength={
                        key === "disclosure"
                          ? 300
                          : key === "values"
                            ? 2000
                            : 1000
                      }
                      value={persona[key] || ""}
                      onChange={(e) => {
                        personaDirty.current = true;
                        setPersona((old) => ({
                          ...old,
                          [key]: e.target.value,
                        }));
                      }}
                    />
                  </Field>
                ))}
                <button
                  disabled={disabled}
                  onClick={() =>
                    mutate(
                      "/persona",
                      {
                        niche: persona.niche || "",
                        tone: persona.tone || "",
                        values: persona.values || "",
                        disclosure: persona.disclosure || "",
                      },
                      "PATCH",
                    )
                  }
                >
                  <Save size={15} />
                  Save persona
                </button>
              </div>
            </details>
          </section>
          <section className="content-editor">
            <div>
              <div className="section-heading">
                <h2>{editing ? "Edit draft" : "New post"}</h2>
              </div>
              <Field label="Photograph">
                <select
                  value={imageId}
                  disabled={!!editing}
                  onChange={(e) => setImageId(e.target.value)}
                >
                  <option value="">Select a saved photograph</option>
                  {images.map((image) => (
                    <option key={image.id} value={image.id}>
                      {image.origin} · {image.prompt.slice(0, 80)} ·{" "}
                      {image.status}
                    </option>
                  ))}
                </select>
              </Field>
              {images.length < imageTotal && (
                <button onClick={moreImages}>
                  <Plus size={15} />
                  Load more photographs
                </button>
              )}
              {images.find((i) => i.id === imageId)?.image && (
                <button
                  className="content-source"
                  onClick={() =>
                    view([images.find((i) => i.id === imageId)!.image])
                  }
                >
                  <img
                    src={images.find((i) => i.id === imageId)!.image.thumb}
                    alt="Selected post photograph"
                  />
                </button>
              )}
            </div>
            <div className="content-editor-fields">
              <Field label="Caption style">
                <select
                  value={captionStyle}
                  onChange={(e) => setCaptionStyle(e.target.value)}
                >
                  {["auto", "concise", "editorial", "conversational"].map(
                    (style) => (
                      <option key={style} value={style}>
                        {style}
                      </option>
                    ),
                  )}
                </select>
              </Field>
              <Field label="Caption brief">
                <input
                  value={brief}
                  onChange={(e) => setBrief(e.target.value)}
                  maxLength={12000}
                />
              </Field>
              <div className="button-row">
                <button
                  disabled={disabled || !imageId}
                  onClick={async () => {
                    const job = await run("caption", {
                      item_id: imageId,
                      details: brief,
                      use_api: useApi,
                      caption_style: captionStyle,
                    });
                    if (job) setCaptionJob(job.id);
                  }}
                >
                  <WandSparkles size={15} />
                  Compose caption
                </button>
                <span className="status">
                  {useApi ? "Gemini" : "Local templates"}
                </span>
                <label className="check">
                  <input
                    type="checkbox"
                    checked={useApi}
                    onChange={(e) => setUseApi(e.target.checked)}
                  />
                  Use Gemini
                </label>
              </div>
              <Field label="Post caption">
                <textarea
                  rows={6}
                  maxLength={2200}
                  value={caption}
                  onChange={(e) => setCaption(e.target.value)}
                />
              </Field>
              <Field label="Planned time">
                <input
                  type="datetime-local"
                  value={planned}
                  onChange={(e) => setPlanned(e.target.value)}
                />
              </Field>
              <Field label="Public JPEG URL (optional)">
                <input
                  type="url"
                  value={publicUrl}
                  onChange={(e) => setPublicUrl(e.target.value)}
                  maxLength={2000}
                />
              </Field>
              <div className="button-row">
                <button
                  className="primary"
                  disabled={disabled || !imageId || !caption.trim()}
                  onClick={async () => {
                    if (
                      await mutate(
                        editing ? "/content/" + editing : "/content",
                        {
                          image_id: imageId,
                          caption,
                          planned_at: planned
                            ? new Date(planned).toISOString()
                            : "",
                          public_url: publicUrl,
                        },
                        editing ? "PATCH" : "POST",
                      )
                    ) {
                      setEditing(null);
                      setCaption("");
                      setPlanned("");
                      setPublicUrl("");
                    }
                  }}
                >
                  <Save size={15} />
                  Save draft
                </button>
                {editing && (
                  <button
                    onClick={() => {
                      setEditing(null);
                      setCaption("");
                      setImageId("");
                      setPlanned("");
                      setPublicUrl("");
                    }}
                  >
                    Cancel edit
                  </button>
                )}
              </div>
            </div>
          </section>
          <section>
            <div className="section-heading">
              <h2>Content calendar</h2>
              <span className="muted">{content.posts.length} posts</span>
              <label className="check">
                <input
                  type="checkbox"
                  aria-label="Runtime scheduling"
                  checked={content.scheduling_enabled}
                  disabled={saving || !content.instagram_configured}
                  onChange={(e) => {
                    const enabled = e.target.checked;
                    if (
                      !enabled ||
                      window.confirm(
                        "Publish approved future posts at their planned time while this notebook is running? Scheduling turns off after a runtime restart; missed posts stay manual.",
                      )
                    )
                      mutate("/scheduling", { confirmed: enabled });
                  }}
                />
                Runtime scheduling
              </label>
            </div>
            <div className="content-posts">
              {[...content.posts]
                .sort(
                  (a, b) =>
                    (a.planned_at
                      ? Date.parse(a.planned_at)
                      : Number.MAX_SAFE_INTEGER) -
                    (b.planned_at
                      ? Date.parse(b.planned_at)
                      : Number.MAX_SAFE_INTEGER),
                )
                .map((post) => (
                  <article key={post.id}>
                    <button
                      className="content-post-image"
                      onClick={() => view([post.image])}
                    >
                      <img src={post.image.thumb} alt="Saved post photograph" />
                    </button>
                    <div>
                      <div className="section-heading">
                        <strong>
                          {post.planned_at
                            ? new Date(post.planned_at).toLocaleString()
                            : "Unscheduled"}
                        </strong>
                        <span className="status">
                          {post.status.replaceAll("_", " ")}
                        </span>
                      </div>
                      <p>{post.caption}</p>
                      <div className="button-row">
                        <a
                          className="button"
                          href={
                            "/api" + base + "/content/" + post.id + "/export"
                          }
                        >
                          <Download size={15} />
                          Export
                        </a>
                        {["draft", "approved", "publish_failed"].includes(
                          post.status,
                        ) && (
                          <button
                            disabled={disabled}
                            onClick={() => {
                              setEditing(post.id);
                              setImageId(post.image_id);
                              setCaption(post.caption);
                              setPublicUrl(post.public_url);
                              setPlanned(
                                post.planned_at
                                  ? new Date(
                                      new Date(post.planned_at).getTime() -
                                        new Date(
                                          post.planned_at,
                                        ).getTimezoneOffset() *
                                          60000,
                                    )
                                      .toISOString()
                                      .slice(0, 16)
                                  : "",
                              );
                              document
                                .querySelector(".content-editor")
                                ?.scrollIntoView({ behavior: "smooth" });
                            }}
                          >
                            Edit
                          </button>
                        )}
                        {post.status === "draft" && (
                          <button
                            disabled={disabled}
                            onClick={() =>
                              window.confirm(
                                "Approve this exact image and caption?\n\n" +
                                  post.caption,
                              ) &&
                              mutate("/content/" + post.id + "/approve", {
                                confirmed: true,
                              })
                            }
                          >
                            <Check size={15} />
                            Approve post
                          </button>
                        )}
                        {content.instagram_configured &&
                          ["approved", "container_created"].includes(
                            post.status,
                          ) && (
                            <button
                              disabled={disabled || !post.public_url}
                              onClick={() =>
                                window.confirm(
                                  "Publish this approved post to Instagram now?",
                                ) && run("publish", { post_id: post.id })
                              }
                            >
                              <Send size={15} />
                              Publish now
                            </button>
                          )}
                        {content.instagram_configured &&
                          ["publishing_unknown", "creating_unknown"].includes(
                            post.status,
                          ) && (
                            <button
                              disabled={disabled}
                              onClick={() => {
                                const media_id = window.prompt(
                                  "Published Instagram media ID",
                                );
                                if (media_id)
                                  run("reconcile", {
                                    post_id: post.id,
                                    media_id,
                                  });
                              }}
                            >
                              Reconcile published post
                            </button>
                          )}
                        {post.permalink && (
                          <a
                            className="button"
                            href={post.permalink}
                            target="_blank"
                            rel="noreferrer"
                          >
                            View post
                          </a>
                        )}
                      </div>
                    </div>
                  </article>
                ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
