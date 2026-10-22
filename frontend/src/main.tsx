import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Camera,
  Users,
  Images,
  Plus,
  Settings2,
  Sun,
  Moon,
  ArrowUp,
  ArrowLeft,
  ArrowRight,
  X,
  Download,
  Expand,
  Trash2,
  ImagePlus,
  Check,
  Loader2,
  Square,
  RefreshCw,
  WandSparkles,
  Search,
  Activity,
  ChevronLeft,
  ChevronRight,
  PanelLeft,
  Copy,
  Undo2,
  Redo2,
  Eraser,
  Paintbrush,
  ScanFace,
  UserRound,
  SlidersHorizontal,
  LogOut,
} from "lucide-react";
import { ReactSketchCanvas, ReactSketchCanvasRef } from "react-sketch-canvas";
import {
  api,
  write,
  upload,
  submit,
  active,
  Avatar,
  Picture,
  Turn,
  GalleryItem,
  Job,
  Config,
} from "./api";
import "./styles.css";

const FLUX = "FLUX.2-klein 4B";
const ModalError = React.createContext("");
function IconButton({
  title,
  children,
  onClick,
  disabled = false,
  className = "",
}: {
  title: string;
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <button
      className={"icon-button " + className}
      title={title}
      aria-label={title}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}
function Modal({
  title,
  children,
  onClose,
  wide = false,
  drawer = false,
}: {
  title: string;
  children: React.ReactNode;
  onClose: () => void;
  wide?: boolean;
  drawer?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const error = React.useContext(ModalError);
  useEffect(() => {
    const el = ref.current!;
    el.showModal();
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      el.close();
      document.body.style.overflow = previous;
    };
  }, []);
  return (
    <dialog
      ref={ref}
      aria-label={title}
      className={"modal" + (wide ? " wide" : "") + (drawer ? " drawer" : "")}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <header>
        <h2>{title}</h2>
        <IconButton title="Close" onClick={onClose}>
          <X />
        </IconButton>
      </header>
      {error && (
        <p className="modal-error" role="alert">
          {error}
        </p>
      )}
      {children}
    </dialog>
  );
}
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
      {children}
    </label>
  );
}
function Range({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (n: number) => void;
}) {
  return (
    <Field label={label}>
      <div className="range-value">
        <input
          aria-label={label}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          style={
            {
              "--fill": `${((value - min) / (max - min)) * 100}%`,
            } as React.CSSProperties
          }
          onChange={(e) => onChange(+e.target.value)}
        />
        <input
          aria-label={label + " value"}
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => {
            const n = +e.target.value;
            if (Number.isFinite(n)) onChange(Math.max(min, Math.min(max, n)));
          }}
        />
      </div>
    </Field>
  );
}
function App() {
  const [config, setConfig] = useState<Config | null>(null),
    [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [boot, setBoot] = useState(true);
  const [theme, setTheme] = useState(
      localStorage.getItem("ffs-theme") || "dark",
    ),
    [section, setSection] = useState<"create" | "avatars">("create"),
    [sidebar, setSidebar] = useState(false);
  const [avatars, setAvatars] = useState<Avatar[]>([]),
    [avatarId, setAvatarId] = useState(
      localStorage.getItem("ffs-avatar") || "",
    ),
    [avatar, setAvatar] = useState<Avatar | null>(null),
    [tab, setTab] = useState("console"),
    [avatarSearch, setAvatarSearch] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]),
    [prompt, setPrompt] = useState(""),
    [extras, setExtras] = useState<Picture[]>([]),
    [canvas, setCanvas] = useState<number | null>(null),
    [uploading, setUploading] = useState(false);
  const [model, setModel] = useState(FLUX),
    [aspect, setAspect] = useState("1024x1024 (1:1)"),
    [steps, setSteps] = useState(4),
    [cfg, setCfg] = useState(1),
    [denoise, setDenoise] = useState(1),
    [seed, setSeed] = useState("0"),
    [quantity, setQuantity] = useState(1),
    [negative, setNegative] = useState(""),
    [maskMode, setMaskMode] = useState("None"),
    [mask, setMask] = useState<Picture | null>(null),
    [mode, setMode] = useState("Identity Strict");
  const [settings, setSettings] = useState(false),
    [jobsOpen, setJobsOpen] = useState(false),
    [jobs, setJobs] = useState<Job[]>([]),
    [sending, setSending] = useState(false),
    [newAvatar, setNewAvatar] = useState(false);
  const [name, setName] = useState(""),
    [creation, setCreation] = useState("Guided"),
    [preset, setPreset] = useState("Studio creator"),
    [details, setDetails] = useState(""),
    [candidateUpload, setCandidateUpload] = useState<Picture | null>(null),
    [guideOptions, setGuideOptions] = useState<Record<string, string>>({});
  const [gallery, setGallery] = useState<GalleryItem[]>([]),
    [total, setTotal] = useState(0),
    [filter, setFilter] = useState("all"),
    [origin, setOrigin] = useState("all"),
    [query, setQuery] = useState(""),
    [batchOpen, setBatchOpen] = useState(false),
    [batchQuantity, setBatchQuantity] = useState(20),
    [batchTheme, setBatchTheme] = useState(""),
    [requireFace, setRequireFace] = useState(false),
    [findings, setFindings] = useState<any>(null),
    [findingsOpen, setFindingsOpen] = useState(false),
    [referenceFilter, setReferenceFilter] = useState("All findings");
  const [viewer, setViewer] = useState<{
      images: Picture[];
      index: number;
      item?: GalleryItem;
    } | null>(null),
    [enhance, setEnhance] = useState<GalleryItem | null>(null),
    [enhanceMethod, setEnhanceMethod] = useState("photo"),
    [scale, setScale] = useState(1),
    [strength, setStrength] = useState(1),
    [correction, setCorrection] = useState<GalleryItem | null>(null),
    [reason, setReason] = useState(""),
    [diagnostics, setDiagnostics] = useState<unknown>(null),
    [maskEditor, setMaskEditor] = useState(false),
    [brush, setBrush] = useState(30),
    [eraser, setEraser] = useState(false);
  const [maskRatio, setMaskRatio] = useState(1);
  const [maskPaths, setMaskPaths] = useState<
    Awaited<ReturnType<ReactSketchCanvasRef["exportPaths"]>>
  >([]);
  const sketch = useRef<ReactSketchCanvasRef>(null),
    input = useRef<HTMLInputElement>(null),
    referenceInput = useRef<HTMLInputElement>(null),
    thread = useRef<HTMLDivElement>(null),
    follow = useRef(true),
    lastJobs = useRef(""),
    requestBusy = useRef(false);
  const busy = jobs.some(active) || sending,
    selectedJob = jobs.find(active),
    isAvatar = section === "avatars",
    canEdit =
      model === FLUX &&
      config?.models
        .find((m) => m.name === model)
        ?.capabilities.includes("img2img"),
    maxExtras = isAvatar ? 2 : 4;
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("ffs-theme", theme);
  }, [theme]);
  async function refresh() {
    const [c, a, h, j] = await Promise.all([
      api<Config>("/config"),
      api<Avatar[]>("/avatars"),
      api<{ turns: Turn[] }>("/chat"),
      api<Job[]>("/jobs"),
    ]);
    setConfig(c);
    setAvatars(a);
    setTurns(h.turns);
    setJobs(j);
    if (avatarId) {
      const data = await api<{ items: GalleryItem[]; total: number }>(
        "/avatars/" +
          avatarId +
          "/gallery?" +
          new URLSearchParams({
            offset: "0",
            limit: String(Math.max(48, gallery.length)),
            status: filter,
            origin,
            query,
          }),
      );
      setGallery(data.items);
      setTotal(data.total);
    }
    if (avatarId) {
      const found = a.find((v) => v.id === avatarId);
      if (found) {
        setAvatar(found);
        setGuideOptions(found.guide.options);
      }
    }
  }
  useEffect(() => {
    (async () => {
      try {
        const token = new URLSearchParams(location.hash.slice(1)).get("access");
        if (token) {
          await write("/session", { token });
          history.replaceState(null, "", location.pathname + location.search);
        }
        await refresh();
      } catch (e) {
        setError(String(e));
      } finally {
        setBoot(false);
      }
    })();
  }, []);
  useEffect(() => {
    let alive = true;
    const timer = setInterval(async () => {
      try {
        const j = await api<Job[]>("/jobs");
        if (!alive) return;
        setJobs(j);
        const stamp = j.map((x) => x.id + ":" + x.status).join();
        if (stamp !== lastJobs.current) {
          await refresh();
          if (j[0]?.status === "failed") setError(j[0].error || j[0].message);
        }
        lastJobs.current = stamp;
        setNotice((value) =>
          value.startsWith("Connection interrupted.") ? "" : value,
        );
        if (
          j.some((x) => active(x) && x.payload.avatar_id === avatarId) &&
          avatarId
        )
          await loadFindings();
      } catch (e) {
        if (alive)
          setNotice(
            "Connection interrupted. Saved jobs are retained; reconnecting.",
          );
      }
    }, 2000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [avatarId, filter, origin, query]);
  useEffect(() => {
    if (avatarId) {
      localStorage.setItem("ffs-avatar", avatarId);
      api<Avatar>("/avatars/" + avatarId)
        .then((a) => {
          setAvatar(a);
          setGuideOptions(a.guide.options);
        })
        .catch((e) => setError(String(e)));
    }
    setExtras([]);
    setCanvas(null);
    setCandidateUpload(null);
  }, [avatarId]);
  useEffect(() => {
    if (isAvatar && avatarId) {
      loadGallery(0);
      loadFindings();
    }
  }, [isAvatar, avatarId, filter, origin, query]);
  useEffect(() => {
    if (tab === "guide" && avatar && !avatar.guide.active)
      setTab(
        avatar.body_locked ? "console" : avatar.face_locked ? "body" : "face",
      );
  }, [avatar?.guide.active, avatar?.id, tab]);
  useEffect(() => {
    if (!maskEditor || canvas === null || !extras[canvas]) return;
    const frame = requestAnimationFrame(() =>
      sketch.current?.loadPaths(maskPaths),
    );
    const image = new Image();
    image.onload = () => setMaskRatio(image.naturalWidth / image.naturalHeight);
    image.src = extras[canvas].url;
    return () => cancelAnimationFrame(frame);
  }, [maskEditor, canvas]);
  useEffect(() => {
    if (follow.current && thread.current)
      thread.current.scrollTop = thread.current.scrollHeight;
  }, [turns, avatar?.chat, selectedJob?.message, tab, section]);
  useEffect(() => {
    if (!viewer) return;
    const key = (event: KeyboardEvent) => {
      if (event.key === "ArrowLeft") moveViewer(-1);
      if (event.key === "ArrowRight") moveViewer(1);
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [viewer]);
  function fail(e: unknown) {
    setError(e instanceof Error ? e.message : String(e));
  }
  function clearMask() {
    setMask(null);
    setMaskPaths([]);
  }
  async function action(value: Record<string, unknown>, close?: () => void) {
    if (requestBusy.current) return;
    requestBusy.current = true;
    setSending(true);
    setError("");
    try {
      const j = await submit(value);
      setJobs((rows) => [j, ...rows.filter((v) => v.id !== j.id)]);
      close?.();
      return j;
    } catch (e) {
      fail(e);
    } finally {
      requestBusy.current = false;
      setSending(false);
    }
  }
  async function loadGallery(offset = 0) {
    if (!avatarId) return;
    try {
      const data = await api<{ items: GalleryItem[]; total: number }>(
        "/avatars/" +
          avatarId +
          "/gallery?" +
          new URLSearchParams({
            offset: String(offset),
            limit: "48",
            status: filter,
            origin,
            query,
          }),
      );
      setGallery((previous) =>
        offset ? [...previous, ...data.items] : data.items,
      );
      setTotal(data.total);
    } catch (e) {
      fail(e);
    }
  }
  async function loadFindings() {
    if (!avatarId) return;
    try {
      setFindings(
        await api(
          "/avatars/" +
            avatarId +
            "/references?status=" +
            encodeURIComponent(referenceFilter),
        ),
      );
    } catch (e) {
      fail(e);
    }
  }
  useEffect(() => {
    if (findingsOpen) loadFindings();
  }, [referenceFilter, findingsOpen]);
  async function addFiles(files: FileList | null, reference = false) {
    if (!files) return;
    setUploading(true);
    try {
      if (reference) {
        const picture = await upload(files[0]);
        setCandidateUpload(picture);
        return;
      }
      const remaining = maxExtras - extras.length;
      if (files.length > remaining)
        setNotice(`Only ${remaining} reference slot(s) remain.`);
      const added: Picture[] = [];
      for (const file of Array.from(files).slice(0, remaining))
        added.push(await upload(file));
      setExtras((old) => [...old, ...added]);
      clearMask();
    } catch (e) {
      fail(e);
    } finally {
      setUploading(false);
      if (input.current) input.current.value = "";
      if (referenceInput.current) referenceInput.current.value = "";
    }
  }
  function removeExtra(index: number) {
    setExtras((old) => old.filter((_, i) => i !== index));
    setCanvas((old) =>
      old === index ? null : old !== null && old > index ? old - 1 : old,
    );
    clearMask();
  }
  function reorder(index: number, direction: number) {
    const next = index + direction;
    if (next < 0 || next >= extras.length) return;
    setExtras((old) => {
      const copy = [...old];
      [copy[index], copy[next]] = [copy[next], copy[index]];
      return copy;
    });
    setCanvas((old) => (old === index ? next : old === next ? index : old));
    clearMask();
  }
  async function send() {
    const value = isAvatar
      ? {
          kind: "console",
          avatar_id: avatarId,
          prompt,
          images: extras.map((i) => i.path),
          aspect,
          steps,
          mode,
        }
      : {
          kind: "generate",
          prompt,
          model,
          aspect,
          steps,
          cfg,
          denoise,
          seed: Number(seed) || 0,
          quantity,
          negative,
          images: extras.map((i) => i.path),
          canvas,
          mask_mode: maskMode,
          mask: mask?.path || null,
        };
    const result = await action(value);
    if (result) {
      setPrompt("");
      setExtras([]);
      setCanvas(null);
      clearMask();
      setMaskMode("None");
      follow.current = true;
    }
  }
  async function create() {
    if (!name.trim()) return;
    setSending(true);
    try {
      const a = await write<Avatar>("/avatars", {
        name,
        mode: creation,
        preset,
      });
      setAvatars((old) => [...old, a]);
      setAvatarId(a.id);
      setAvatar(a);
      setSection("avatars");
      setTab(creation === "Guided" ? "guide" : "face");
      setNewAvatar(false);
      setName("");
      setSidebar(false);
    } catch (e) {
      fail(e);
    } finally {
      setSending(false);
    }
  }
  async function guideAction(kind: string) {
    try {
      await write("/avatars/" + avatarId + "/guide/" + kind);
      await refresh();
    } catch (e) {
      fail(e);
    }
  }
  function view(images: Picture[], index = 0, item?: GalleryItem) {
    setViewer({ images, index, item });
  }
  function moveViewer(direction: number) {
    setViewer((old) => {
      if (!old) return old;
      const index = Math.max(
        0,
        Math.min(old.images.length - 1, old.index + direction),
      );
      const item = old.item
        ? gallery.find((v) => v.image.path === old.images[index].path)
        : undefined;
      return { ...old, index, item };
    });
  }
  function useInput(picture: Picture) {
    setSection("create");
    setExtras([picture]);
    setCanvas(0);
    setModel(FLUX);
    setSteps(4);
    setMaskMode("None");
    clearMask();
    setViewer(null);
    setNotice("Image set as canvas.");
    follow.current = true;
  }
  function modelChanged(value: string) {
    setModel(value);
    const defaults = config?.models.find((row) => row.name === value)?.defaults;
    setSteps(defaults?.steps || 4);
    setCfg(defaults?.cfg || 1);
    if (value !== FLUX) {
      setExtras([]);
      setCanvas(null);
      setMaskMode("None");
      clearMask();
    }
  }
  const displayTurns = isAvatar ? avatar?.chat || [] : turns;
  function chat() {
    return (
      <div className="chat-layout">
        <div
          ref={thread}
          className="thread"
          onScroll={() => {
            const el = thread.current;
            if (el)
              follow.current =
                el.scrollHeight - el.scrollTop - el.clientHeight < 160;
          }}
        >
          {!displayTurns.length && (
            <div className="empty-chat">
              <Camera size={30} />
              <h2>
                {isAvatar ? `Create with ${avatar?.name}` : "Create an image"}
              </h2>
            </div>
          )}
          {displayTurns.map((turn) => (
            <article key={turn.id} className={"turn " + turn.role}>
              <div className="turn-role">
                {turn.role === "user"
                  ? "You"
                  : isAvatar
                    ? avatar?.name
                    : "Studio"}
                {turn.seed && <span>Seed {turn.seed}</span>}
              </div>
              {turn.text && <p>{turn.text}</p>}
              {turn.images?.length > 0 && (
                <div
                  className={
                    "turn-images " + (turn.role === "user" ? "references" : "")
                  }
                >
                  {turn.images.filter(Boolean).map((image, index) => (
                    <figure key={image.path}>
                      <button
                        className="image-open"
                        onClick={() => view(turn.images, index)}
                      >
                        <img
                          src={turn.role === "user" ? image.thumb : image.url}
                          width={image.width}
                          height={image.height}
                          onLoad={() => {
                            if (follow.current && thread.current)
                              thread.current.scrollTop =
                                thread.current.scrollHeight;
                          }}
                          loading="lazy"
                          alt={
                            turn.role === "user"
                              ? "Attached reference"
                              : "Generated image"
                          }
                        />
                      </button>
                      {turn.role !== "user" && (
                        <figcaption>
                          <IconButton
                            title="Full screen"
                            onClick={() => view(turn.images, index)}
                          >
                            <Expand />
                          </IconButton>
                          <a
                            className="icon-button"
                            title="Download"
                            aria-label="Download"
                            href={image.url + "?download=true"}
                          >
                            <Download />
                          </a>
                          <IconButton
                            title="Use as canvas"
                            onClick={() => useInput(image)}
                          >
                            <ImagePlus />
                          </IconButton>
                          {!isAvatar && turn.recipe && (
                            <IconButton
                              title="Regenerate"
                              disabled={busy}
                              onClick={() =>
                                action({
                                  ...turn.recipe,
                                  kind: "generate",
                                  seed: 0,
                                })
                              }
                            >
                              <RefreshCw />
                            </IconButton>
                          )}
                        </figcaption>
                      )}
                    </figure>
                  ))}
                </div>
              )}
            </article>
          ))}
          {selectedJob && (
            <div className="generation-status" role="status">
              <Loader2 className="spin" />
              <div>
                <strong>{selectedJob.message}</strong>
                <small>
                  {selectedJob.kind} / {selectedJob.status}
                </small>
              </div>
              <IconButton
                title="Stop operation"
                onClick={() =>
                  write("/jobs/" + selectedJob.id + "/cancel")
                    .then(refresh)
                    .catch(fail)
                }
              >
                <Square />
              </IconButton>
            </div>
          )}
        </div>
        <footer className="composer">
          <div className="composer-inner">
            {isAvatar && avatar && (
              <div className="identity-strip">
                {[avatar.face, avatar.body].map(
                  (pic, i) =>
                    pic && (
                      <button key={i} onClick={() => view([pic])}>
                        <img
                          src={pic.thumb}
                          alt={i ? "Body anchor" : "Face anchor"}
                        />
                        <span>
                          {i ? "Body" : "Face"}
                          <Check size={12} />
                        </span>
                      </button>
                    ),
                )}
                <span className="muted">{mode}</span>
              </div>
            )}
            {extras.length > 0 && (
              <div className="attachments">
                {extras.map((pic, i) => (
                  <div
                    className={"attachment " + (canvas === i ? "canvas" : "")}
                    key={pic.path}
                  >
                    <button
                      className="attachment-image"
                      onClick={() => view(extras, i)}
                    >
                      <img src={pic.thumb} alt={"Reference " + (i + 1)} />
                    </button>
                    <div className="attachment-actions">
                      <IconButton
                        title="Remove image"
                        onClick={() => removeExtra(i)}
                      >
                        <X />
                      </IconButton>
                      <span>{i + 1}</span>
                      <IconButton
                        title="Move left"
                        disabled={!i}
                        onClick={() => reorder(i, -1)}
                      >
                        <ArrowLeft />
                      </IconButton>
                      <IconButton
                        title="Move right"
                        disabled={i === extras.length - 1}
                        onClick={() => reorder(i, 1)}
                      >
                        <ArrowRight />
                      </IconButton>
                    </div>
                    {!isAvatar && (
                      <button
                        className="canvas-toggle"
                        aria-pressed={canvas === i}
                        onClick={() => {
                          setCanvas(canvas === i ? null : i);
                          clearMask();
                        }}
                      >
                        {canvas === i ? (
                          <Check size={12} />
                        ) : (
                          <Images size={12} />
                        )}{" "}
                        {canvas === i ? "Canvas" : "Reference"}
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
            <div className="prompt-box">
              <textarea
                aria-label="Image prompt"
                placeholder={
                  isAvatar
                    ? "Describe the next photograph..."
                    : "Describe your image or edit..."
                }
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (
                    e.key === "Enter" &&
                    (e.ctrlKey || e.metaKey) &&
                    !busy &&
                    prompt.trim()
                  ) {
                    e.preventDefault();
                    send();
                  }
                }}
              />
              <div className="composer-tools">
                <div>
                  {(isAvatar || canEdit) && (
                    <IconButton
                      title={`Add reference (${extras.length}/${maxExtras})`}
                      disabled={uploading || extras.length >= maxExtras || busy}
                      onClick={() => input.current?.click()}
                    >
                      {uploading ? <Loader2 className="spin" /> : <Plus />}
                    </IconButton>
                  )}
                  <button className="quiet" onClick={() => setSettings(true)}>
                    <SlidersHorizontal size={15} />
                    {steps} steps
                    <span className="desktop-only">
                      {" "}
                      / {aspect.split(" ")[0]}
                    </span>
                  </button>
                </div>
                <button
                  className="send-button"
                  aria-label="Generate image"
                  title="Generate image"
                  onClick={send}
                  disabled={
                    busy ||
                    uploading ||
                    !prompt.trim() ||
                    (isAvatar && (!avatar?.face_locked || !avatar?.body_locked))
                  }
                >
                  {busy ? <Loader2 className="spin" /> : <ArrowUp />}
                </button>
              </div>
            </div>
          </div>
        </footer>
      </div>
    );
  }
  function referenceStep(kind: "face" | "body") {
    if (!avatar) return null;
    const pic =
      candidateUpload ||
      avatar[(kind + "_candidate") as "face_candidate"] ||
      avatar[kind];
    const locked = avatar[(kind + "_locked") as "face_locked"];
    return (
      <div className="reference-workspace">
        <div className={"reference-stage " + kind}>
          {pic ? (
            <button onClick={() => view([pic])}>
              <img src={pic.url} alt={kind + " reference"} />
            </button>
          ) : (
            <div className="placeholder">
              {kind === "face" ? (
                <ScanFace size={40} />
              ) : (
                <UserRound size={40} />
              )}
            </div>
          )}
        </div>
        <div className="reference-controls">
          <h2>{kind === "face" ? "Face reference" : "Body reference"}</h2>
          <span className={"status " + (locked ? "passed" : "")}>
            {locked ? "Confirmed" : "Awaiting confirmation"}
          </span>
          <Field label="Details">
            <textarea
              rows={6}
              value={details}
              onChange={(e) => setDetails(e.target.value)}
              placeholder={
                kind === "face"
                  ? "Hair, eyes, facial structure, expression..."
                  : "Build, proportions, clothing, posture..."
              }
            />
          </Field>
          <div className="button-row">
            <button
              className="primary"
              disabled={busy || (kind === "body" && !avatar.face_locked)}
              onClick={() =>
                action(
                  {
                    kind: "reference",
                    avatar_id: avatarId,
                    reference_kind: kind,
                    details,
                  },
                  () => setCandidateUpload(null),
                )
              }
            >
              <WandSparkles size={16} />
              Generate
            </button>
            <button
              disabled={busy}
              onClick={() => referenceInput.current?.click()}
            >
              <ImagePlus size={16} />
              Upload
            </button>
          </div>
          <button
            className="primary"
            disabled={busy || !pic}
            onClick={() =>
              action(
                {
                  kind: "confirm",
                  avatar_id: avatarId,
                  reference_kind: kind,
                  image: pic?.path,
                },
                () => setCandidateUpload(null),
              )
            }
          >
            <Check size={16} />
            Confirm & analyze {kind}
          </button>
          {locked && (
            <button
              onClick={() => {
                setTab(kind === "face" ? "body" : "console");
                setDetails("");
                setCandidateUpload(null);
              }}
            >
              Next
              <ArrowRight size={16} />
            </button>
          )}
          <details>
            <summary>Identity specifications</summary>
            <pre>
              {JSON.stringify(
                avatar[(kind + "_specs") as "face_specs"],
                null,
                2,
              )}
            </pre>
          </details>
        </div>
      </div>
    );
  }
  function galleryView() {
    return (
      <div className="gallery-view">
        <div className="view-toolbar">
          <div>
            <h2>
              Gallery <span className="muted">{total}</span>
            </h2>
          </div>
          <div className="button-row">
            <button
              onClick={() => {
                loadFindings();
                setFindingsOpen(true);
              }}
            >
              <Activity size={16} />
              Pipeline
            </button>
            <button
              className="primary"
              disabled={busy || !avatar?.body_locked}
              onClick={() => setBatchOpen(true)}
            >
              <Plus size={16} />
              Create batch
            </button>
          </div>
        </div>
        <div className="gallery-filters">
          <div className="search-box">
            <Search size={16} />
            <input
              aria-label="Search gallery"
              placeholder="Search photographs"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
          <select
            aria-label="Gallery status"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          >
            {["all", "passed", "needs_review", "unchecked", "no_change"].map(
              (s) => (
                <option key={s} value={s}>
                  {s.replaceAll("_", " ")}
                </option>
              ),
            )}
          </select>
          <select
            aria-label="Gallery source"
            value={origin}
            onChange={(e) => setOrigin(e.target.value)}
          >
            {["all", "console", "guided", "auto", "enhancement", "legacy"].map(
              (s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ),
            )}
          </select>
        </div>
        {!gallery.length ? (
          <div className="empty">
            <Images size={32} />
            <h3>No images here yet</h3>
          </div>
        ) : (
          <div className="gallery-grid">
            {gallery.map((item, index) => (
              <article className="gallery-tile" key={item.id}>
                <button
                  className="gallery-image"
                  onClick={() =>
                    view(
                      gallery.map((v) => v.image),
                      index,
                      item,
                    )
                  }
                >
                  <img
                    src={item.image.thumb}
                    loading="lazy"
                    alt={item.prompt || "Avatar photograph"}
                  />
                </button>
                <div className="tile-overlay">
                  <span className={"status " + item.status}>
                    {item.status?.replaceAll("_", " ")}
                  </span>
                  <IconButton
                    title="Reconstruct photographic detail"
                    disabled={busy}
                    onClick={() => setEnhance(item)}
                  >
                    <WandSparkles />
                  </IconButton>
                </div>
                <div className="tile-caption">
                  <span>{item.origin}</span>
                  <a
                    title="Download image"
                    aria-label="Download image"
                    href={item.image.url + "?download=true"}
                  >
                    <Download size={15} />
                  </a>
                </div>
              </article>
            ))}
          </div>
        )}
        {gallery.length < total && (
          <button
            className="load-more"
            onClick={() => loadGallery(gallery.length)}
          >
            Load more ({total - gallery.length})
          </button>
        )}
      </div>
    );
  }
  if (boot)
    return (
      <div className="boot">
        <Loader2 className="spin" />
        Opening studio...
      </div>
    );
  if (!config)
    return (
      <div className="boot">
        <h1>FreeFakeStudio</h1>
        <p role="alert">{error}</p>
        <button onClick={() => location.reload()}>Reconnect</button>
      </div>
    );
  return (
    <ModalError.Provider value={error}>
      <div className="app-shell">
        <aside className={"navigation " + (sidebar ? "open" : "")}>
          <div className="brand">
            <Camera />
            <strong>FreeFakeStudio</strong>
            <IconButton
              title="Close navigation"
              className="mobile-only"
              onClick={() => setSidebar(false)}
            >
              <X />
            </IconButton>
          </div>
          <nav>
            <button
              className={section === "create" ? "selected" : ""}
              onClick={() => {
                setSection("create");
                setSidebar(false);
                setExtras([]);
                setCanvas(null);
              }}
            >
              <Camera size={18} />
              Create
            </button>
            <button
              className={isAvatar ? "selected" : ""}
              onClick={() => {
                setSection("avatars");
                setSidebar(false);
                setExtras([]);
                setCanvas(null);
              }}
            >
              <Users size={18} />
              Avatar Studio
            </button>
          </nav>
          <div className="library-header">
            <span>AVATARS</span>
            <IconButton title="New avatar" onClick={() => setNewAvatar(true)}>
              <Plus />
            </IconButton>
          </div>
          <div className="search-box compact">
            <Search size={14} />
            <input
              aria-label="Find avatar"
              placeholder="Find avatar"
              value={avatarSearch}
              onChange={(e) => setAvatarSearch(e.target.value)}
            />
          </div>
          <div className="avatar-list">
            {avatars
              .filter((a) =>
                a.name.toLowerCase().includes(avatarSearch.toLowerCase()),
              )
              .map((a) => (
                <button
                  className={isAvatar && avatarId === a.id ? "selected" : ""}
                  key={a.id}
                  onClick={() => {
                    setAvatarId(a.id);
                    setSection("avatars");
                    setTab(
                      a.guide.active
                        ? "guide"
                        : a.body_locked
                          ? "console"
                          : a.face_locked
                            ? "body"
                            : "face",
                    );
                    setSidebar(false);
                    setDetails("");
                  }}
                >
                  {a.face ? (
                    <img src={a.face.thumb} alt="" />
                  ) : (
                    <UserRound size={20} />
                  )}
                  <span>
                    {a.name}
                    <small>
                      {a.body_locked
                        ? "Ready"
                        : a.face_locked
                          ? "Body setup"
                          : "Face setup"}
                    </small>
                  </span>
                </button>
              ))}
          </div>
          <div className="navigation-bottom">
            <span className="runtime-dot" />
            {config.mock ? "Local mock preview" : config.provider}
            <button
              className="quiet"
              onClick={() =>
                api("/diagnostics").then(setDiagnostics).catch(fail)
              }
            >
              Diagnostics
            </button>
          </div>
        </aside>
        {sidebar && (
          <button
            aria-label="Close navigation"
            className="nav-shade"
            onClick={() => setSidebar(false)}
          />
        )}
        <main>
          <header className="app-header">
            <div className="header-title">
              <IconButton
                title="Open navigation"
                className="mobile-only"
                onClick={() => setSidebar(true)}
              >
                <PanelLeft />
              </IconButton>
              <h1>
                {isAvatar ? avatar?.name || "Avatar Studio" : "Image Studio"}
              </h1>
              {!isAvatar && (
                <select
                  aria-label="Model"
                  value={model}
                  disabled={busy}
                  onChange={(e) => modelChanged(e.target.value)}
                >
                  {config.models.map((m) => (
                    <option
                      key={m.name}
                      value={m.name}
                      disabled={m.status === "missing"}
                    >
                      {m.name}
                      {m.status === "missing" ? " (not installed)" : ""}
                    </option>
                  ))}
                </select>
              )}
            </div>
            <div className="header-actions">
              <IconButton
                title="Job activity"
                onClick={() => setJobsOpen(true)}
              >
                <Activity />
                {jobs.some(active) && <span className="activity-dot" />}
              </IconButton>
              <IconButton
                title={theme === "dark" ? "Light theme" : "Dark theme"}
                onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              >
                {theme === "dark" ? <Sun /> : <Moon />}
              </IconButton>
              <IconButton title="Settings" onClick={() => setSettings(true)}>
                <Settings2 />
              </IconButton>
            </div>
          </header>
          {error && (
            <div className="alert" role="alert">
              <span>{error}</span>
              <IconButton title="Dismiss error" onClick={() => setError("")}>
                <X />
              </IconButton>
            </div>
          )}
          {notice && (
            <div className="notice" role="status">
              <span>{notice}</span>
              <IconButton title="Dismiss notice" onClick={() => setNotice("")}>
                <X />
              </IconButton>
            </div>
          )}
          {isAvatar && avatar && (
            <div className="avatar-tabs" role="tablist">
              {[
                ...(avatar.guide.active ? ["guide"] : []),
                "face",
                "body",
                "console",
                "gallery",
              ].map((value) => (
                <button
                  role="tab"
                  aria-selected={tab === value}
                  className={tab === value ? "selected" : ""}
                  key={value}
                  onClick={() => {
                    setTab(value);
                    setCandidateUpload(null);
                    setDetails("");
                  }}
                >
                  {value === "guide"
                    ? "Guided setup"
                    : value[0].toUpperCase() + value.slice(1)}
                  {["face", "body"].includes(value) &&
                    avatar[(value + "_locked") as "face_locked"] && (
                      <Check size={13} />
                    )}
                </button>
              ))}
              <button
                className="rename-button"
                title="Rename avatar"
                onClick={() => {
                  const name = window.prompt("Avatar name", avatar.name);
                  if (name?.trim())
                    write("/avatars/" + avatarId, { name }, "PATCH")
                      .then(refresh)
                      .catch(fail);
                }}
              >
                <Settings2 size={15} />
              </button>
            </div>
          )}
          {isAvatar && tab !== "console" && selectedJob && (
            <div className="operation-bar" role="status">
              <Loader2 className="spin" size={16} />
              <span>{selectedJob.message}</span>
              <IconButton
                title="Stop operation"
                onClick={() =>
                  write("/jobs/" + selectedJob.id + "/cancel")
                    .then(refresh)
                    .catch(fail)
                }
              >
                <Square />
              </IconButton>
            </div>
          )}
          {!isAvatar ? (
            chat()
          ) : !avatar ? (
            <div className="empty">
              <Users size={36} />
              <h2>Avatar Studio</h2>
              <button className="primary" onClick={() => setNewAvatar(true)}>
                <Plus size={17} />
                New avatar
              </button>
            </div>
          ) : tab === "console" ? (
            chat()
          ) : tab === "gallery" ? (
            galleryView()
          ) : tab === "guide" ? (
            <div className="guide-workspace">
              <div className="view-toolbar">
                <div>
                  <p className="eyebrow">GUIDED SETUP</p>
                  <h2>{avatar.guide.stage.replaceAll("_", " ")}</h2>
                </div>
                <button disabled={busy} onClick={() => guideAction("custom")}>
                  Switch to custom
                </button>
              </div>
              <div className="guide-preview">
                {(avatar.guide.face_draft || avatar.face) && (
                  <button
                    onClick={() =>
                      view([avatar.guide.face_draft || avatar.face!])
                    }
                  >
                    <img
                      src={(avatar.guide.face_draft || avatar.face)!.url}
                      alt="Face draft"
                    />
                  </button>
                )}
                {(avatar.guide.body_draft || avatar.body) && (
                  <button
                    onClick={() =>
                      view([avatar.guide.body_draft || avatar.body!])
                    }
                  >
                    <img
                      src={(avatar.guide.body_draft || avatar.body)!.url}
                      alt="Body draft"
                    />
                  </button>
                )}
              </div>
              <div className="guide-options">
                {Object.entries(guideOptions).map(([key, value]) => (
                  <Field key={key} label={key.replaceAll("_", " ")}>
                    <textarea
                      rows={3}
                      value={value}
                      onChange={(e) =>
                        setGuideOptions((old) => ({
                          ...old,
                          [key]: e.target.value,
                        }))
                      }
                    />
                  </Field>
                ))}
              </div>
              <div className="button-row">
                <button disabled={busy} onClick={() => guideAction("back")}>
                  <ArrowLeft size={16} />
                  Back
                </button>
                <button
                  disabled={busy}
                  onClick={() => guideAction("regenerate")}
                >
                  <RefreshCw size={16} />
                  Regenerate draft
                </button>
                <button
                  disabled={busy}
                  onClick={() => referenceInput.current?.click()}
                >
                  <ImagePlus size={16} />
                  Upload instead
                </button>
                <button
                  className="primary"
                  disabled={busy}
                  onClick={() =>
                    action(
                      {
                        kind: "guide",
                        avatar_id: avatarId,
                        options: guideOptions,
                        image: candidateUpload?.path || null,
                      },
                      () => setCandidateUpload(null),
                    )
                  }
                >
                  {busy ? (
                    <Loader2 className="spin" size={16} />
                  ) : (
                    <ArrowRight size={16} />
                  )}{" "}
                  {avatar.guide.next_label}
                </button>
              </div>
              {candidateUpload && (
                <div className="upload-ready">
                  <img src={candidateUpload.thumb} alt="Uploaded replacement" />
                  <span>Replacement attached</span>
                  <IconButton
                    title="Remove replacement"
                    onClick={() => setCandidateUpload(null)}
                  >
                    <X />
                  </IconButton>
                </div>
              )}
            </div>
          ) : (
            referenceStep(tab as "face" | "body")
          )}
        </main>
        <input
          ref={input}
          className="hidden"
          type="file"
          accept="image/png,image/jpeg,image/webp"
          multiple
          onChange={(e) => addFiles(e.target.files)}
        />
        <input
          ref={referenceInput}
          className="hidden"
          type="file"
          accept="image/png,image/jpeg,image/webp"
          onChange={(e) => addFiles(e.target.files, true)}
        />
        {settings && (
          <Modal
            title="Generation settings"
            drawer
            onClose={() => setSettings(false)}
          >
            <div className="settings-content">
              <Field label="Aspect ratio">
                <select
                  value={aspect}
                  onChange={(e) => setAspect(e.target.value)}
                >
                  {config.aspects.map((v) => (
                    <option key={v}>{v}</option>
                  ))}
                </select>
              </Field>
              <Range
                label="Steps"
                value={steps}
                min={1}
                max={
                  isAvatar
                    ? 50
                    : config.models.find((row) => row.name === model)?.defaults
                        .max_steps || 50
                }
                onChange={setSteps}
              />
              {isAvatar ? (
                <Field label="Reference mode">
                  <select
                    value={mode}
                    onChange={(e) => setMode(e.target.value)}
                  >
                    {config.modes.map((v) => (
                      <option key={v}>{v}</option>
                    ))}
                  </select>
                </Field>
              ) : (
                <>
                  <Range
                    label="Images"
                    value={quantity}
                    min={1}
                    max={8}
                    onChange={setQuantity}
                  />
                  <Field label="Seed">
                    <input
                      type="number"
                      min={0}
                      max={Number.MAX_SAFE_INTEGER}
                      value={seed}
                      onChange={(e) => setSeed(e.target.value)}
                    />
                  </Field>
                  <Range
                    label="Guidance"
                    value={cfg}
                    min={0}
                    max={20}
                    step={0.1}
                    onChange={setCfg}
                  />
                  {canEdit && extras.length > 0 && canvas !== null && (
                    <>
                      <Range
                        label="Edit strength"
                        value={denoise}
                        min={0.05}
                        max={1}
                        step={0.05}
                        onChange={setDenoise}
                      />
                      <Field label="Mask">
                        <select
                          value={maskMode}
                          onChange={(e) => {
                            setMaskMode(e.target.value);
                            clearMask();
                          }}
                        >
                          {config.mask_modes.map((v) => (
                            <option key={v}>{v}</option>
                          ))}
                        </select>
                      </Field>
                      {maskMode === "Manual Paint" ? (
                        <button onClick={() => setMaskEditor(true)}>
                          <Paintbrush size={16} />
                          {mask ? "Edit painted mask" : "Paint mask"}
                        </button>
                      ) : (
                        maskMode !== "None" && (
                          <button
                            disabled={busy}
                            onClick={() =>
                              action({
                                kind: "mask_preview",
                                image: extras[canvas].path,
                                mask_mode: maskMode,
                                prompt,
                              })
                            }
                          >
                            <ScanFace size={16} />
                            Preview mask
                          </button>
                        )
                      )}
                      {mask && (
                        <button onClick={() => view([mask])}>
                          <img
                            className="mask-thumb"
                            src={mask.thumb}
                            alt="Painted mask"
                          />
                        </button>
                      )}
                    </>
                  )}
                  <Field label="Negative prompt">
                    <textarea
                      value={negative}
                      onChange={(e) => setNegative(e.target.value)}
                      rows={3}
                    />
                  </Field>
                </>
              )}
              {(isAvatar || model === FLUX) && (
                <Field label="FLUX text encoder">
                  <select
                    disabled={busy}
                    value={config.encoder}
                    onChange={(e) =>
                      action({ kind: "encoder", encoder: e.target.value })
                    }
                  >
                    {config.encoder_choices.map((v) => (
                      <option key={v}>{v}</option>
                    ))}
                  </select>
                </Field>
              )}
              <div className="button-row">
                <button
                  disabled={busy}
                  onClick={() => action({ kind: "unload" })}
                >
                  <LogOut size={16} />
                  Unload model
                </button>
                {!isAvatar && (
                  <button
                    disabled={busy}
                    onClick={() => {
                      if (
                        confirm(
                          "Archive this conversation and start a new one?",
                        )
                      )
                        api("/chat", { method: "DELETE" })
                          .then(refresh)
                          .catch(fail);
                    }}
                  >
                    <Trash2 size={16} />
                    New conversation
                  </button>
                )}
              </div>
            </div>
          </Modal>
        )}
        {newAvatar && (
          <Modal title="New avatar" onClose={() => setNewAvatar(false)}>
            <div className="settings-content">
              <Field label="Name">
                <input
                  autoFocus
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  maxLength={80}
                />
              </Field>
              <div className="segmented">
                {["Guided", "Custom"].map((v) => (
                  <button
                    key={v}
                    aria-pressed={creation === v}
                    onClick={() => setCreation(v)}
                  >
                    {v}
                  </button>
                ))}
              </div>
              <Field label="Preset">
                <select
                  value={preset}
                  onChange={(e) => setPreset(e.target.value)}
                >
                  {config.presets.map((v) => (
                    <option key={v}>{v}</option>
                  ))}
                </select>
              </Field>
              <button
                className="primary"
                disabled={sending || !name.trim()}
                onClick={create}
              >
                <Plus size={16} />
                Create avatar
              </button>
            </div>
          </Modal>
        )}
        {jobsOpen && (
          <Modal title="Activity" onClose={() => setJobsOpen(false)} wide>
            <div className="jobs-list">
              {!jobs.length && <p className="muted">No jobs yet.</p>}
              {jobs.map((job) => (
                <article key={job.id}>
                  <div>
                    <span className={"status " + job.status}>{job.status}</span>
                    <strong>{job.kind}</strong>
                    <p>{job.message}</p>
                    {job.error && <p className="error-text">{job.error}</p>}
                    {job.sync_error && (
                      <p className="error-text">Drive sync: {job.sync_error}</p>
                    )}
                  </div>
                  {active(job) ? (
                    <button
                      disabled={job.cancel_requested}
                      onClick={() =>
                        write("/jobs/" + job.id + "/cancel")
                          .then(refresh)
                          .catch(fail)
                      }
                    >
                      <Square size={14} />
                      Stop
                    </button>
                  ) : (
                    ["failed", "interrupted", "cancelled"].includes(
                      job.status,
                    ) && (
                      <button
                        onClick={() =>
                          write("/jobs/" + job.id + "/resume")
                            .then(refresh)
                            .catch(fail)
                        }
                      >
                        <RefreshCw size={14} />
                        Resume
                      </button>
                    )
                  )}
                  {job.kind === "mask_preview" &&
                    Boolean(job.result?.image) && (
                      <button
                        onClick={() =>
                          view([
                            {
                              path: String(job.result!.image),
                              url: "/api/files/" + job.result!.image,
                              thumb: "/api/thumbs/" + job.result!.image,
                            },
                          ])
                        }
                      >
                        View mask
                      </button>
                    )}
                </article>
              ))}
            </div>
          </Modal>
        )}
        {viewer && (
          <Modal
            title={viewer.item ? "Photograph" : "Image"}
            onClose={() => setViewer(null)}
            wide
          >
            <div className="viewer">
              <div className="viewer-image">
                <img
                  src={viewer.images[viewer.index].url}
                  width={viewer.images[viewer.index].width}
                  height={viewer.images[viewer.index].height}
                  alt="Full-size photograph"
                />
              </div>
              <div className="viewer-toolbar">
                <div className="button-row">
                  <IconButton
                    title="Previous image"
                    disabled={!viewer.index}
                    onClick={() => moveViewer(-1)}
                  >
                    <ChevronLeft />
                  </IconButton>
                  <span>
                    {viewer.index + 1} / {viewer.images.length}
                  </span>
                  <IconButton
                    title="Next image"
                    disabled={viewer.index === viewer.images.length - 1}
                    onClick={() => moveViewer(1)}
                  >
                    <ChevronRight />
                  </IconButton>
                </div>
                <div className="button-row">
                  <a
                    className="button"
                    href={viewer.images[viewer.index].url + "?download=true"}
                  >
                    <Download size={16} />
                    Download
                  </a>
                  <button onClick={() => useInput(viewer.images[viewer.index])}>
                    <ImagePlus size={16} />
                    Use as canvas
                  </button>
                  {viewer.item && (
                    <>
                      <button
                        disabled={busy}
                        onClick={() => {
                          setEnhance(viewer.item!);
                          setViewer(null);
                        }}
                      >
                        <WandSparkles size={16} />
                        Reconstruct
                      </button>
                      {viewer.item.regenerable && (
                        <button
                          disabled={busy}
                          onClick={() => {
                            setCorrection(viewer.item!);
                            setReason("");
                            setViewer(null);
                          }}
                        >
                          <RefreshCw size={16} />
                          Correct
                        </button>
                      )}
                    </>
                  )}
                </div>
              </div>
              {viewer.item && (
                <details className="image-details">
                  <summary>Generation details</summary>
                  <p>{viewer.item.prompt}</p>
                  <pre>
                    {JSON.stringify(
                      {
                        status: viewer.item.status,
                        origin: viewer.item.origin,
                        seed: viewer.item.seed,
                        passes: viewer.item.processing_passes,
                        diagnostics: viewer.item.diagnostics,
                        validation: viewer.item.validation,
                      },
                      null,
                      2,
                    )}
                  </pre>
                </details>
              )}
              {viewer.item?.source && (
                <div className="comparison">
                  <figure>
                    <img src={viewer.item.source.url} alt="Original" />
                    <figcaption>Original</figcaption>
                  </figure>
                  <figure>
                    <img src={viewer.item.image.url} alt="Reconstruction" />
                    <figcaption>Reconstruction</figcaption>
                  </figure>
                </div>
              )}
            </div>
          </Modal>
        )}
        {enhance && (
          <Modal
            title="Photographic reconstruction"
            onClose={() => setEnhance(null)}
          >
            <div className="settings-content">
              <img
                className="enhance-preview"
                src={enhance.image.thumb}
                alt="Source photograph"
              />
              <Field label="Processing">
                <select
                  value={enhanceMethod}
                  onChange={(e) => setEnhanceMethod(e.target.value)}
                >
                  <option value="photo">One reconstruction pass</option>
                  <option value="hd_tiles">HD tiled reconstruction</option>
                </select>
              </Field>
              <Field label="Output size">
                <select
                  value={scale}
                  onChange={(e) => setScale(+e.target.value)}
                >
                  <option value={1}>Original dimensions</option>
                  <option value={2}>2x dimensions</option>
                </select>
              </Field>
              <Range
                label="Reconstruction strength"
                value={strength}
                min={0.35}
                max={1}
                step={0.05}
                onChange={setStrength}
              />
              <button
                className="primary"
                disabled={busy}
                onClick={() =>
                  action(
                    {
                      kind: "enhance",
                      avatar_id: avatarId,
                      item_id: enhance.id,
                      method: enhanceMethod,
                      scale,
                      strength,
                      steps: 4,
                    },
                    () => setEnhance(null),
                  )
                }
              >
                <WandSparkles size={16} />
                Reconstruct image
              </button>
            </div>
          </Modal>
        )}
        {correction && (
          <Modal
            title="Correct this photograph"
            onClose={() => setCorrection(null)}
          >
            <div className="settings-content">
              <Field label="What needs correcting?">
                <textarea
                  rows={5}
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                />
              </Field>
              <button
                className="primary"
                disabled={busy || !reason.trim()}
                onClick={() =>
                  action(
                    {
                      kind: "regenerate",
                      avatar_id: avatarId,
                      item_id: correction.id,
                      reason,
                    },
                    () => setCorrection(null),
                  )
                }
              >
                <RefreshCw size={16} />
                Regenerate
              </button>
            </div>
          </Modal>
        )}
        {batchOpen && (
          <Modal title="Gallery batch" onClose={() => setBatchOpen(false)}>
            <div className="settings-content">
              <Range
                label="Photographs"
                value={batchQuantity}
                min={1}
                max={20}
                onChange={setBatchQuantity}
              />
              <Field label="Theme">
                <textarea
                  rows={4}
                  value={batchTheme}
                  onChange={(e) => setBatchTheme(e.target.value)}
                  placeholder="Wardrobe, locations, mood..."
                />
              </Field>
              <label className="check-field">
                <input
                  type="checkbox"
                  checked={requireFace}
                  onChange={(e) => setRequireFace(e.target.checked)}
                />
                Require a visible face in references
              </label>
              <button
                className="primary"
                disabled={busy}
                onClick={() =>
                  action(
                    {
                      kind: "gallery",
                      avatar_id: avatarId,
                      quantity: batchQuantity,
                      aspect,
                      theme: batchTheme,
                      require_face: requireFace,
                    },
                    () => {
                      setBatchOpen(false);
                      setFindingsOpen(true);
                    },
                  )
                }
              >
                <ArrowRight size={16} />
                {findings?.job ? "Continue saved batch" : "Start batch"}
              </button>
              <button
                disabled={busy}
                onClick={async () => {
                  if (
                    !confirm(
                      "Start a fresh search? Saved reference files and generated images will be retained.",
                    )
                  )
                    return;
                  try {
                    await write("/avatars/" + avatarId + "/gallery/new-search");
                    await loadFindings();
                  } catch (e) {
                    fail(e);
                  }
                }}
              >
                <Search size={16} />
                New search
              </button>
            </div>
          </Modal>
        )}
        {findingsOpen && (
          <Modal
            title="Reference pipeline"
            onClose={() => setFindingsOpen(false)}
            wide
          >
            <div className="pipeline">
              <div className="pipeline-counts">
                {Object.entries(findings?.counts || {}).map(([key, value]) => (
                  <div key={key}>
                    <strong>{String(value)}</strong>
                    <span>{key.replaceAll("_", " ")}</span>
                  </div>
                ))}
              </div>
              <div className="view-toolbar">
                <span>
                  {findings?.job?.message ||
                    findings?.job?.phase ||
                    "No active batch"}
                </span>
                <div className="button-row">
                  <IconButton title="Refresh pipeline" onClick={loadFindings}>
                    <RefreshCw />
                  </IconButton>
                  <button
                    disabled={busy}
                    onClick={() => {
                      setFindingsOpen(false);
                      setBatchOpen(true);
                    }}
                  >
                    <ArrowRight size={15} />
                    Continue batch
                  </button>
                </div>
              </div>
              {findings?.error && (
                <p className="error-text">{findings.error}</p>
              )}
              <Field label="Findings">
                <select
                  value={referenceFilter}
                  onChange={(e) => setReferenceFilter(e.target.value)}
                >
                  {[
                    "All findings",
                    "Approved",
                    "Rejected",
                    "Awaiting checks",
                  ].map((v) => (
                    <option key={v}>{v}</option>
                  ))}
                </select>
              </Field>
              <div className="reference-grid">
                {findings?.items?.map((row: any) => (
                  <article key={row.id}>
                    {row.image && (
                      <button onClick={() => view([row.image])}>
                        <img
                          src={row.image.thumb}
                          alt="Collected reference"
                          loading="lazy"
                        />
                      </button>
                    )}
                    <span className={"status " + row.state}>{row.state}</span>
                    <p>{row.reason}</p>
                    {row.source_url?.startsWith("https://") && (
                      <a href={row.source_url} target="_blank" rel="noreferrer">
                        Source
                      </a>
                    )}
                  </article>
                ))}
              </div>
              <details>
                <summary>Pipeline events</summary>
                <pre>{JSON.stringify(findings?.events || [], null, 2)}</pre>
              </details>
            </div>
          </Modal>
        )}
        {diagnostics !== null && (
          <Modal title="Diagnostics" onClose={() => setDiagnostics(null)} wide>
            <div className="settings-content">
              <button
                onClick={() =>
                  navigator.clipboard
                    .writeText(JSON.stringify(diagnostics, null, 2))
                    .then(() => setNotice("Diagnostics copied."))
                    .catch(fail)
                }
              >
                <Copy size={16} />
                Copy report
              </button>
              <pre>{JSON.stringify(diagnostics, null, 2)}</pre>
            </div>
          </Modal>
        )}
        {maskEditor && canvas !== null && extras[canvas] && (
          <Modal
            title="Paint edit mask"
            onClose={() => setMaskEditor(false)}
            wide
          >
            <div className="settings-content">
              <div
                className="mask-canvas"
                style={{
                  aspectRatio: maskRatio,
                  width: `min(100%, ${maskRatio * 55}dvh)`,
                  height: "auto",
                }}
              >
                <ReactSketchCanvas
                  ref={sketch}
                  onChange={setMaskPaths}
                  backgroundImage={extras[canvas].url}
                  exportWithBackgroundImage={false}
                  canvasColor="transparent"
                  strokeColor="white"
                  strokeWidth={brush}
                  eraserWidth={brush}
                  width="100%"
                  height="100%"
                  preserveBackgroundImageAspectRatio="xMidYMid meet"
                />
              </div>
              <div className="button-row">
                <IconButton
                  title="Paint"
                  onClick={() => {
                    setEraser(false);
                    sketch.current?.eraseMode(false);
                  }}
                  className={!eraser ? "selected" : ""}
                >
                  <Paintbrush />
                </IconButton>
                <IconButton
                  title="Erase"
                  onClick={() => {
                    setEraser(true);
                    sketch.current?.eraseMode(true);
                  }}
                  className={eraser ? "selected" : ""}
                >
                  <Eraser />
                </IconButton>
                <IconButton title="Undo" onClick={() => sketch.current?.undo()}>
                  <Undo2 />
                </IconButton>
                <IconButton title="Redo" onClick={() => sketch.current?.redo()}>
                  <Redo2 />
                </IconButton>
                <IconButton
                  title="Clear mask"
                  onClick={() => sketch.current?.clearCanvas()}
                >
                  <Trash2 />
                </IconButton>
              </div>
              <Range
                label="Brush size"
                value={brush}
                min={4}
                max={100}
                onChange={setBrush}
              />
              <button
                className="primary"
                disabled={
                  !maskPaths.some(
                    (path) => path.drawMode && path.paths.length > 0,
                  )
                }
                onClick={async () => {
                  try {
                    const data = await sketch.current!.exportImage("png");
                    const blob = await (await fetch(data)).blob();
                    setMask(
                      await upload(
                        new File([blob], "mask.png", { type: "image/png" }),
                      ),
                    );
                    setMaskMode("Manual Paint");
                    setMaskEditor(false);
                  } catch (e) {
                    fail(e);
                  }
                }}
              >
                <Check size={16} />
                Use mask
              </button>
            </div>
          </Modal>
        )}
      </div>
    </ModalError.Provider>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
