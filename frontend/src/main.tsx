import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, useLocation, useNavigate } from "react-router-dom";
import {
  studioRoute,
  chatPath,
  newChatPath,
  avatarPath,
  AvatarTab,
  GallerySection,
} from "./routes";
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
  MessageSquare,
  Pencil,
  Archive,
  Columns2,
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
  ChatSummary,
  Conversation,
  Resource,
  GalleryCounts,
  GalleryPage,
  ResourcePage,
  GalleryBatch,
} from "./api";
import "./styles.css";
import { InfluencerWorkspace } from "./InfluencerWorkspace";
import { GalleryBatches } from "./GalleryBatches";

const FLUX = "FLUX.2-klein 4B";
const GALLERY_SECTIONS = [
  { id: "generations", label: "Generations", icon: Images },
  { id: "enhancements", label: "Enhancements", icon: WandSparkles },
  { id: "resources", label: "Resources", icon: ImagePlus },
  { id: "review", label: "Needs Review", icon: ScanFace },
] as const;
type ComposerDraft = {
  prompt: string;
  extras: Picture[];
  canvas: number | null;
  mask: Picture | null;
  maskMode: string;
  maskPaths: Awaited<ReturnType<ReactSketchCanvasRef["exportPaths"]>>;
};
type EnhancementTarget = {
  image: Picture;
  avatar_id: string | null;
  chat_id?: string;
  item_id?: string;
};
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
  className = "",
}: {
  title: string;
  children: React.ReactNode;
  onClose: () => void;
  wide?: boolean;
  drawer?: boolean;
  className?: string;
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
      className={
        "modal" +
        (wide ? " wide" : "") +
        (drawer ? " drawer" : "") +
        (className ? " " + className : "")
      }
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
  const location = useLocation();
  const batchFilter = new URLSearchParams(location.search).get("batch") || "";
  const navigate = useNavigate();
  const routeKey = useRef(location.key);
  routeKey.current = location.key;
  const route = studioRoute(location.pathname);
  const section =
    route.kind === "avatar" || route.kind === "library" ? "avatars" : "create";
  const avatarId = route.avatarId || "";
  const tab = route.tab || "console";
  const gallerySection = route.gallerySection || "generations";
  const requestedChatId = route.chatId;
  const isChatRoute =
    route.kind === "main" ||
    (route.kind === "avatar" && route.tab === "console");
  const isDraftRoute = Boolean(route.draft);
  const [routeError, setRouteError] = useState("");
  const [config, setConfig] = useState<Config | null>(null),
    [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [boot, setBoot] = useState(true);
  const [theme, setTheme] = useState(
      localStorage.getItem("ffs-theme") || "dark",
    ),
    [sidebar, setSidebar] = useState(false);
  const [avatars, setAvatars] = useState<Avatar[]>([]),
    [storedAvatar, setAvatar] = useState<Avatar | null>(null),
    [avatarSearch, setAvatarSearch] = useState("");
  const avatar = storedAvatar?.id === avatarId ? storedAvatar : null;
  const [turns, setTurns] = useState<Turn[]>([]),
    [prompt, setPrompt] = useState(""),
    [extras, setExtras] = useState<Picture[]>([]),
    [canvas, setCanvas] = useState<number | null>(null),
    [uploading, setUploading] = useState(false);
  const [chats, setChats] = useState<ChatSummary[]>([]),
    [chatId, setChatId] = useState(""),
    [chatLoading, setChatLoading] = useState(true),
    [chatSearch, setChatSearch] = useState("");
  const [renameChat, setRenameChat] = useState<ChatSummary | null>(null),
    [chatName, setChatName] = useState("");
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
    [mode, setMode] = useState("Identity Strict"),
    [faceView, setFaceView] = useState("auto"),
    [bodyView, setBodyView] = useState("auto");
  const [settings, setSettings] = useState(false),
    [savingAppSettings, setSavingAppSettings] = useState(false),
    [jobsOpen, setJobsOpen] = useState(false),
    [jobs, setJobs] = useState<Job[]>([]),
    [sending, setSending] = useState(false),
    [newAvatar, setNewAvatar] = useState(false);
  const appSettingsEpoch = useRef(0),
    appSettingsPending = useRef(false);
  const [name, setName] = useState(""),
    [creation, setCreation] = useState("Guided"),
    [preset, setPreset] = useState("Studio creator"),
    [details, setDetails] = useState(""),
    [candidateUpload, setCandidateUpload] = useState<Picture | null>(null),
    [guideOptions, setGuideOptions] = useState<Record<string, string>>({});
  const [gallery, setGallery] = useState<GalleryItem[]>([]),
    [galleryCounts, setGalleryCounts] = useState<GalleryCounts>({
      generations: 0,
      enhancements: 0,
      resources: 0,
      review: 0,
    }),
    [galleryLoading, setGalleryLoading] = useState(false),
    [resources, setResources] = useState<Resource[]>([]),
    [resourceTotal, setResourceTotal] = useState(0),
    [resourceCounts, setResourceCounts] = useState<Record<string, number>>({}),
    [resourceFilter, setResourceFilter] = useState("all"),
    [resourceKind, setResourceKind] = useState("all"),
    [total, setTotal] = useState(0),
    [filter, setFilter] = useState("all"),
    [origin, setOrigin] = useState("all"),
    [query, setQuery] = useState(""),
    [batchOpen, setBatchOpen] = useState(false),
    [batchManagerOpen, setBatchManagerOpen] = useState(false),
    [batchName, setBatchName] = useState(""),
    [batches, setBatches] = useState<GalleryBatch[]>([]),
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
      chat_id?: string;
      avatar_id?: string | null;
      source?: Picture | null;
    } | null>(null),
    [enhance, setEnhance] = useState<EnhancementTarget | null>(null),
    [comparing, setComparing] = useState(false),
    [comparisonRegion, setComparisonRegion] = useState("full"),
    [enhanceMethod, setEnhanceMethod] = useState("photo"),
    [enhanceSampling, setEnhanceSampling] = useState("reference"),
    [enhanceProfile, setEnhanceProfile] = useState("auto"),
    [enhancementPrompt, setEnhancementPrompt] = useState(""),
    [enhanceInstructions, setEnhanceInstructions] = useState(""),
    [preserveTone, setPreserveTone] = useState(true),
    [scale, setScale] = useState(1),
    [strength, setStrength] = useState(0.65),
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
    isAvatar = section === "avatars",
    canEdit =
      model === FLUX &&
      config?.models
        .find((m) => m.name === model)
        ?.capabilities.includes("img2img"),
    maxExtras = isAvatar ? 2 : 4;
  const scope = isAvatar && avatarId ? avatarId : null;
  const galleryRequest = useRef(0);
  const galleryLoaded = useRef({ generations: 0, resources: 0 });
  galleryLoaded.current = {
    generations: gallery.length,
    resources: resources.length,
  };
  const galleryContext = useRef({
    avatarId,
    batchFilter,
    gallerySection,
    filter,
    origin,
    query,
    resourceFilter,
    resourceKind,
  });
  galleryContext.current = {
    avatarId,
    batchFilter,
    gallerySection,
    filter,
    origin,
    query,
    resourceFilter,
    resourceKind,
  };
  const reconstructionPrompt =
    enhancementPrompt;
  const selectedEnhancementProfile = config?.enhancement_profiles?.find((profile) => profile.id === enhanceProfile);
  const customEnhancement = enhanceProfile === "custom";
  useLayoutEffect(() => {
    if (!enhance) return;
    setEnhancementPrompt("");
    setEnhanceInstructions("");
    setEnhanceProfile("auto");
    setEnhanceSampling("reference");
    setEnhanceMethod("photo");
    setScale(1);
  }, [enhance?.image.path]);
  const viewerSource = viewer?.item?.source || viewer?.source;
  const viewerDiagnostics =
    viewer?.item?.diagnostics ||
    turns.find((turn) =>
      turn.images.some(
        (pic) => pic.path === viewer?.images[viewer.index]?.path,
      ),
    )?.diagnostics;
  const detailRegions = (viewerDiagnostics?.detail_regions as
    { label: string; box: number[] }[] | undefined) || [
    { label: "Center detail", box: [0.25, 0.25, 0.75, 0.75] },
  ];
  const detailEvidence = viewerDiagnostics?.detail_evidence as
    | {
        advisory?: string;
        source_detail_energy?: number;
        result_detail_energy?: number;
      }
    | undefined;
  const viewerPlan = viewerDiagnostics?.enhancement_plan as
    | {
        automatic: boolean;
        prompt: string;
        focus_rules?: string[];
        profile_name?: string;
        selection?: string;
        analysis: Record<string, unknown>;
      }
    | undefined;
  const textureReview = viewerDiagnostics?.texture_review as
    { status: string; reasons?: string[] } | undefined;
  const enhancementReview = viewerDiagnostics?.enhancement_review as
    { status: string; reasons?: string[] } | undefined;
  function comparisonUrl(pic: Picture) {
    const region =
      comparisonRegion === "full"
        ? undefined
        : detailRegions[Number(comparisonRegion)];
    return region
      ? pic.thumb +
          (pic.thumb.includes("?") ? "&" : "?") +
          "size=800&region=" +
          region.box.join(",")
      : pic.url;
  }
  const chatContext = useRef<{ scope: string | null; id: string }>({
    scope,
    id: chatId,
  });
  chatContext.current = { scope, id: chatId };
  const chatEpoch = useRef(0);
  const drafts = useRef<Record<string, ComposerDraft>>({}),
    pendingCanvas = useRef<Picture | null>(null);
  const draftOwner = useRef<{ scope: string | null; id: string } | null>(null);
  const conversationPending =
    chatLoading ||
    Boolean(isDraftRoute && chatId) ||
    Boolean(requestedChatId && requestedChatId !== chatId) ||
    Boolean(isChatRoute && scope !== draftOwner.current?.scope);
  const selectedJob = jobs.find(
    (job) =>
      active(job) &&
      (job.payload.chat_id
        ? job.payload.chat_id === chatId &&
          (job.payload.avatar_id || null) === scope
        : (job.payload.avatar_id || null) === scope),
  );
  const scopeQuery = (value: string | null) =>
    value ? "?avatar_id=" + encodeURIComponent(value) : "";
  const chatKey = (value: string | null) => "ffs-chat-" + (value || "main");
  function rememberDraft() {
    if (draftOwner.current?.id === chatId && draftOwner.current.scope === scope)
      drafts.current[(scope || "main") + ":" + chatId] = {
        prompt,
        extras,
        canvas,
        mask,
        maskMode,
        maskPaths,
      };
  }
  function restoreDraft(value: string | null, id: string) {
    draftOwner.current = { scope: value, id };
    const draft = drafts.current[(value || "main") + ":" + id];
    setPrompt(draft?.prompt || "");
    setExtras(draft?.extras || []);
    setCanvas(draft?.canvas ?? null);
    setMask(draft?.mask || null);
    setMaskMode(draft?.maskMode || "None");
    setMaskPaths(draft?.maskPaths || []);
    if (value === null && pendingCanvas.current) {
      setExtras([pendingCanvas.current]);
      setCanvas(0);
      clearMask();
      setMaskMode("None");
      pendingCanvas.current = null;
    }
    follow.current = true;
  }
  useEffect(() => {
    rememberDraft();
  }, [prompt, extras, canvas, mask, maskMode, maskPaths, chatId, scope]);
  function go(path: string, replace = false) {
    rememberDraft();
    navigate(path, { replace });
  }
  function setTab(value: string, replace = false) {
    if (!avatarId) return;
    go(
      value === "console" &&
        chatId &&
        !chatLoading &&
        draftOwner.current?.scope === avatarId
        ? chatPath(chatId, avatarId)
        : avatarPath(avatarId, value as AvatarTab, gallerySection),
      replace,
    );
  }
  useEffect(() => {
    setRouteError("");
    setError("");
    setSidebar(false);
    setViewer(null);
    setEnhance(null);
    setCorrection(null);
    setSettings(false);
    setJobsOpen(false);
    setNewAvatar(false);
    setMaskEditor(false);
    setFindingsOpen(false);
    setBatchManagerOpen(false);
    setBatchOpen(false);
    setDetails("");
    setCandidateUpload(null);
  }, [location.pathname]);
  useEffect(() => {setBatches([]);}, [avatarId]);
  async function refreshConversations() {
    const current = { ...chatContext.current };
    const rows = await api<ChatSummary[]>("/chats" + scopeQuery(current.scope));
    const row = current.id
      ? await api<Conversation>(
          "/chats/" + current.id + scopeQuery(current.scope),
        )
      : null;
    if (
      chatContext.current.scope === current.scope &&
      chatContext.current.id === current.id
    ) {
      setChats(rows);
      if (row) setTurns(row.turns);
    }
  }
  async function selectChat(id: string, targetScope = scope) {
    go(chatPath(id, targetScope));
  }
  function startChat(targetScope: string | null = scope) {
    setSidebar(false);
    setSettings(false);
    setError("");
    if (isDraftRoute && scope === targetScope) return;
    localStorage.setItem(chatKey(targetScope), "__draft");
    go(newChatPath(targetScope));
  }
  function avatarLibrary() {
    go("/avatars");
  }
  function openAvatar(value: Avatar) {
    rememberDraft();
    setAvatar(value);
    go(
      avatarPath(
        value.id,
        value.guide.active
          ? "guide"
          : value.body_locked
            ? "console"
            : value.face_locked
              ? "body"
              : "face",
      ),
    );
    setSidebar(false);
    setDetails("");
    setGallery([]);
    setResources([]);
    setGalleryCounts({
      generations: 0,
      enhancements: 0,
      resources: 0,
      review: 0,
    });
    setResourceTotal(0);
    setTotal(0);
    setSteps(4);
  }
  useEffect(() => {
    if (boot || route.kind === "notfound") return;
    const epoch = ++chatEpoch.current;
    setChatLoading(true);
    setChats([]);
    setChatId("");
    setTurns([]);
    setChatSearch("");
    (async () => {
      try {
        const rows = await api<ChatSummary[]>("/chats" + scopeQuery(scope));
        if (epoch !== chatEpoch.current) return;
        const saved = localStorage.getItem(chatKey(scope));
        if (
          !requestedChatId &&
          (isDraftRoute || saved === "__draft" || !rows.length)
        ) {
          setChats(rows);
          restoreDraft(scope, "");
          localStorage.setItem(chatKey(scope), "__draft");
          if (isChatRoute && location.pathname !== newChatPath(scope))
            navigate(newChatPath(scope), { replace: true });
          return;
        }
        const id =
          requestedChatId ||
          rows.find((row) => row.id === saved)?.id ||
          rows[0].id;
        const row = await api<Conversation>("/chats/" + id + scopeQuery(scope));
        if (epoch === chatEpoch.current) {
          if (!row.turns.length && isChatRoute) {
            navigate(newChatPath(scope), { replace: true });
            return;
          }
          setChats(rows);
          setChatId(id);
          setTurns(row.turns);
          restoreDraft(scope, id);
          localStorage.setItem(chatKey(scope), id);
          if (isChatRoute && !requestedChatId) {
            navigate(chatPath(id, scope), { replace: true });
          }
        }
      } catch (e) {
        if (epoch === chatEpoch.current) {
          setRouteError(
            requestedChatId
              ? "This conversation is unavailable, archived, or belongs to another avatar."
              : "This workspace is unavailable.",
          );
          fail(e);
        }
      } finally {
        if (epoch === chatEpoch.current) setChatLoading(false);
      }
    })();
    return () => {
      if (chatEpoch.current === epoch) ++chatEpoch.current;
    };
  }, [scope, requestedChatId, boot, route.kind, isChatRoute, isDraftRoute]);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("ffs-theme", theme);
  }, [theme]);
  async function refresh() {
    const settingsEpoch = appSettingsEpoch.current;
    const [c, a, j] = await Promise.all([
      api<Config>("/config"),
      api<Avatar[]>("/avatars"),
      api<Job[]>("/jobs"),
    ]);
    setConfig((current) =>
      current &&
      (appSettingsPending.current || settingsEpoch !== appSettingsEpoch.current)
        ? { ...c, app_settings: current.app_settings }
        : c,
    );
    setAvatars(a);
    setJobs(j);
    await refreshConversations();
    if (avatarId && chatContext.current.scope === avatarId)
      await loadGallery(0, true);
    if (avatarId && chatContext.current.scope === avatarId) await loadBatches();
    if (avatarId && chatContext.current.scope === avatarId) {
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
          navigate(location.pathname + location.search, { replace: true });
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
    if (boot) return;
    let alive = true;
    let refreshing = false,
      needsRefresh = false,
      lastFindings = 0,
      lastEvent = Date.now(),
      fallback = false,
      nextPoll = 0;
    const update = async () => {
      needsRefresh = true;
      if (refreshing) return;
      refreshing = true;
      while (alive && needsRefresh) {
        needsRefresh = false;
        try {
          await refresh();
          if (avatarId) await loadFindings();
        } catch (e) {
          if (alive)
            setNotice(
              "Connection interrupted. Saved jobs are retained; reconnecting.",
            );
        }
      }
      refreshing = false;
    };
    const consume = (j: Job[]) => {
      if (!alive) return;
      lastEvent = Date.now();
      setJobs(j);
      const stamp = j.map((row) => row.id + ":" + row.status).join();
      if (stamp !== lastJobs.current) {
        lastJobs.current = stamp;
        update();
        if (j[0]?.status === "failed") setError(j[0].error || j[0].message);
      }
      setNotice((value) =>
        value.startsWith("Connection interrupted.") ? "" : value,
      );
      if (
        avatarId &&
        j.some((row) => active(row) && row.payload.avatar_id === avatarId) &&
        Date.now() - lastFindings > 5000
      ) {
        lastFindings = Date.now();
        loadFindings();
      }
    };
    const poll = async () => {
      try {
        const rows = await api<Job[]>("/jobs");
        consume(rows);
        nextPoll = Date.now() + (rows.some(active) ? 8000 : 45000);
      } catch (e) {
        nextPoll = Date.now() + 15000;
        if (alive)
          setNotice(
            "Connection interrupted. Saved jobs are retained; reconnecting.",
          );
      }
    };
    const source = new EventSource("/api/activity/events");
    source.onmessage = (event) => {
      try {
        consume(JSON.parse(event.data));
      } catch {
        source.close();
        fallback = true;
        poll();
      }
    };
    source.addEventListener("heartbeat", () => {
      lastEvent = Date.now();
    });
    source.onerror = () => {
      source.close();
      if (!fallback) {
        fallback = true;
        poll();
      }
    };
    const timer = setInterval(() => {
      if (!fallback && Date.now() - lastEvent > 45000) {
        source.close();
        fallback = true;
      }
      if (fallback && Date.now() >= nextPoll) {
        nextPoll = Date.now() + 60000;
        poll();
      }
    }, 3000);
    const focus = () => {
      refreshConversations().catch(fail);
      if (fallback) poll();
    };
    window.addEventListener("focus", focus);
    return () => {
      alive = false;
      clearInterval(timer);
      source.close();
      window.removeEventListener("focus", focus);
    };
  }, [
    avatarId,
    filter,
    origin,
    query,
    gallerySection,
    resourceFilter,
    resourceKind,
    chatId,
    batchFilter,
    boot,
  ]);
  useEffect(() => {
    let cancelled = false;
    setAvatar(null);
    if (avatarId && !boot) {
      localStorage.setItem("ffs-avatar", avatarId);
      api<Avatar>("/avatars/" + avatarId)
        .then((a) => {
          if (cancelled) return;
          setAvatar(a);
          setGuideOptions(a.guide.options);
        })
        .catch((e) => {
          if (!cancelled) {
            setRouteError("This avatar could not be loaded.");
            fail(e);
          }
        });
    }
    setExtras([]);
    setCanvas(null);
    setCandidateUpload(null);
    return () => {
      cancelled = true;
    };
  }, [avatarId, boot]);
  useEffect(() => {
    setFilter("all");
    setOrigin("all");
    setQuery("");
    setResourceFilter("all");
    setResourceKind("all");
    setGallery([]);
    setResources([]);
  }, [avatarId, gallerySection, batchFilter]);
  useEffect(() => {
    if (!boot && isAvatar && avatarId) {
      loadGallery(0);
      loadFindings();
      loadBatches();
    }
  }, [
    isAvatar,
    avatarId,
    filter,
    origin,
    query,
    gallerySection,
    batchFilter,
    resourceFilter,
    resourceKind,
    boot,
  ]);
  useEffect(() => {
    if (!boot && avatar?.id === avatarId) {
      if (!route.tab || (tab === "guide" && !avatar.guide.active))
        setTab(
          avatar.guide.active
            ? "guide"
            : avatar.body_locked
              ? "console"
              : avatar.face_locked
                ? "body"
                : "face",
          true,
        );
      else if (tab === "gallery" && !route.gallerySection)
        navigate(avatarPath(avatarId, "gallery"), { replace: true });
    }
  }, [
    avatar?.guide.active,
    avatar?.id,
    avatarId,
    route.tab,
    route.gallerySection,
    tab,
    boot,
  ]);
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
  }, [turns, selectedJob?.message, tab, section]);
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
  async function loadGallery(offset = 0, preserve = false) {
    const context = galleryContext.current;
    if (!context.avatarId) return;
    const key = JSON.stringify(context);
    const request = ++galleryRequest.current;
    const current = () =>
      request === galleryRequest.current &&
      key === JSON.stringify(galleryContext.current);
    const pageLimit = preserve
      ? Math.min(
          100,
          Math.max(
            48,
            context.gallerySection === "resources"
              ? galleryLoaded.current.resources
              : galleryLoaded.current.generations,
          ),
        )
      : 48;
    setGalleryLoading(true);
    try {
      const [data, references] = await Promise.all([
        api<GalleryPage>(
          "/avatars/" +
            context.avatarId +
            "/gallery?" +
            new URLSearchParams({
              offset:
                context.gallerySection === "resources" ? "0" : String(offset),
              limit:
                context.gallerySection === "resources"
                  ? "1"
                  : String(pageLimit),
              category:
                context.gallerySection === "resources"
                  ? "all"
                  : context.gallerySection,
              status: context.filter,
              origin: context.origin,
              query: context.query,
              ...(context.batchFilter ? {batch_id: context.batchFilter} : {}),
            }),
        ),
        context.gallerySection === "resources"
          ? api<ResourcePage>(
              "/avatars/" +
                context.avatarId +
                "/resources?" +
                new URLSearchParams({
                  offset: String(offset),
                  limit: String(pageLimit),
                  status: context.resourceFilter,
                  kind: context.resourceKind,
                  query: context.query,
                  ...(context.batchFilter ? {batch_id: context.batchFilter} : {}),
                }),
            )
          : Promise.resolve(null),
      ]);
      if (!current()) return;
      setGalleryCounts(data.counts);
      if (references) {
        setResources((previous) =>
          offset ? [...previous, ...references.items] : references.items,
        );
        setResourceTotal(references.total);
        setResourceCounts(references.counts);
      } else {
        setGallery((previous) =>
          offset ? [...previous, ...data.items] : data.items,
        );
        setTotal(data.total);
      }
    } catch (e) {
      if (current()) fail(e);
    } finally {
      if (current()) setGalleryLoading(false);
    }
  }
  function changeGallerySection(value: GallerySection) {
    if (value === gallerySection) return;
    go(avatarPath(avatarId, "gallery", value) + (batchFilter ? "?batch=" + encodeURIComponent(batchFilter) : ""));
    setFilter("all");
    setOrigin("all");
    setQuery("");
    setGallery([]);
    setResources([]);
    setResourceFilter("all");
    setResourceKind("all");
    setGalleryLoading(true);
  }
  async function loadFindings() {
    if (!avatarId || !batchFilter) {setFindings(null); return;}
    const target = avatarId;
    const selected = batchFilter;
    try {
      const data = await api(
          "/avatars/" +
            avatarId +
            "/references?status=" +
            encodeURIComponent(referenceFilter) + (selected ? "&batch_id=" + encodeURIComponent(selected) : ""),
        );
      if (galleryContext.current.avatarId === target && galleryContext.current.batchFilter === selected) setFindings(data);
    } catch (e) {
      fail(e);
    }
  }
  async function loadBatches() {
    if (!avatarId) return;
    const target = avatarId;
    try {
      const data = await api<{items: GalleryBatch[]}>("/avatars/" + target + "/gallery/batches");
      if (galleryContext.current.avatarId === target) setBatches(data.items);
    } catch (e) { fail(e); }
  }
  function selectBatch(id: string) {
    setFindings(null);
    go(avatarPath(avatarId, "gallery", gallerySection) + (id ? "?batch=" + encodeURIComponent(id) : ""));
  }
  async function resumeBatch(id: string) {
    const job = await action({kind: "gallery", avatar_id: avatarId, batch_id: id});
    if (job) { selectBatch(id); setBatchManagerOpen(false); setFindingsOpen(true); }
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
    if (conversationPending || (!chatId && !isDraftRoute)) return;
    const startedOn = routeKey.current;
    const value = isAvatar
      ? {
          kind: "console",
          avatar_id: avatarId,
          chat_id: chatId || null,
          new_chat: isDraftRoute,
          prompt,
          images: extras.map((i) => i.path),
          roles: extras.map(
            (i) =>
              i.role ||
              {
                "Outfit Focus": "Outfit",
                "Pose Focus": "Pose",
                "Scene Focus": "Scene",
                "Group Image": "Other person",
              }[mode] ||
              "Outfit",
          ),
          aspect,
          steps,
          mode,
          face_view: faceView,
          body_view: bodyView,
        }
      : {
          kind: "generate",
          chat_id: chatId || null,
          new_chat: isDraftRoute,
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
      delete drafts.current[(scope || "main") + ":" + chatId];
      if (
        chatContext.current.scope !== scope ||
        chatContext.current.id !== chatId ||
        routeKey.current !== startedOn
      )
        return;
      setPrompt("");
      setExtras([]);
      setCanvas(null);
      clearMask();
      setMaskMode("None");
      follow.current = true;
      if (isDraftRoute && result.payload.chat_id) {
        const id = String(result.payload.chat_id);
        localStorage.setItem(chatKey(scope), id);
        navigate(chatPath(id, scope), { replace: true });
      }
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
      openAvatar(a);
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
  function view(
    images: Picture[],
    index = 0,
    item?: GalleryItem,
    generated = false,
  ) {
    setComparing(false);
    setComparisonRegion("full");
    setViewer({
      images,
      index,
      item,
      avatar_id: scope,
      chat_id: generated ? chatId : undefined,
      source: generated
        ? turns.find((turn) =>
            turn.images.some((image) => image.path === images[index].path),
          )?.source
        : undefined,
    });
  }
  function moveViewer(direction: number) {
    setComparing(false);
    setComparisonRegion("full");
    setViewer((old) => {
      if (!old) return old;
      const index = Math.max(
        0,
        Math.min(old.images.length - 1, old.index + direction),
      );
      const item = old.item
        ? gallery.find((v) => v.image.path === old.images[index].path)
        : undefined;
      const source = old.chat_id
        ? turns.find((turn) =>
            turn.images.some((image) => image.path === old.images[index].path),
          )?.source
        : undefined;
      return { ...old, index, item, source };
    });
  }
  function useInput(picture: Picture) {
    rememberDraft();
    if (scope) pendingCanvas.current = picture;
    go(
      localStorage.getItem(chatKey(null))
        ? chatPath(localStorage.getItem(chatKey(null))!)
        : "/chats",
    );
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
  const displayTurns = conversationPending ? [] : turns;
  function chat() {
    return (
      <div
        className="chat-layout"
        data-route={location.pathname}
        aria-busy={conversationPending}
      >
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
          {conversationPending ? (
            <div className="empty-chat" role="status">
              <Loader2 className="spin" />
              Loading conversation...
            </div>
          ) : (
            !displayTurns.length && (
              <div className="empty-chat">
                <Camera size={30} />
                <h2>
                  {isAvatar ? `Create with ${avatar?.name}` : "Create an image"}
                </h2>
              </div>
            )
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
                        onClick={() =>
                          view(
                            turn.images,
                            index,
                            undefined,
                            turn.role === "assistant",
                          )
                        }
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
                            onClick={() =>
                              view(turn.images, index, undefined, true)
                            }
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
                          <button
                            className="enhance-action"
                            disabled={busy || conversationPending}
                            onClick={() =>
                              setEnhance({
                                image,
                                avatar_id: scope,
                                chat_id: chatId,
                              })
                            }
                          >
                            <WandSparkles size={16} />
                            Enhance
                          </button>
                          {!isAvatar && turn.recipe && (
                            <IconButton
                              title="Regenerate"
                              disabled={busy}
                              onClick={() =>
                                action({
                                  ...turn.recipe,
                                  kind: "generate",
                                  chat_id: chatId,
                                  new_chat: false,
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
                    {isAvatar && (
                      <select
                        aria-label={"Reference role " + (i + 1)}
                        value={
                          pic.role ||
                          {
                            "Outfit Focus": "Outfit",
                            "Pose Focus": "Pose",
                            "Scene Focus": "Scene",
                            "Group Image": "Other person",
                          }[mode] ||
                          "Outfit"
                        }
                        onChange={(e) =>
                          setExtras((old) =>
                            old.map((item, index) =>
                              index === i
                                ? {
                                    ...item,
                                    role: e.target.value as Picture["role"],
                                  }
                                : item,
                            ),
                          )
                        }
                      >
                        {["Outfit", "Pose", "Scene", "Other person"].map(
                          (role) => (
                            <option key={role}>{role}</option>
                          ),
                        )}
                      </select>
                    )}
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
                disabled={conversationPending || sending}
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
                      disabled={
                        conversationPending ||
                        uploading ||
                        extras.length >= maxExtras ||
                        busy
                      }
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
                    conversationPending ||
                    (!chatId && !isDraftRoute) ||
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
    const isResources = gallerySection === "resources";
    const count = isResources ? resourceTotal : total;
    const loaded = isResources ? resources.length : gallery.length;
    return (
      <div className="gallery-view">
        <div className="view-toolbar">
          <div>
            <h2>
              Gallery <span className="muted">{count}</span>
            </h2>
          </div>
          <div className="button-row">
            <button onClick={() => {loadBatches(); setBatchManagerOpen(true);}}><Columns2 size={16} /> Batches</button>
            <button
              onClick={() => {
                if (!batchFilter) {loadBatches(); setBatchManagerOpen(true); return;}
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
              onClick={() => {setBatchName(""); setBatchOpen(true);}}
            >
              <Plus size={16} />
              Create batch
            </button>
          </div>
        </div>
        <Field label="Gallery batch">
          <select value={batchFilter} onChange={(e) => selectBatch(e.target.value)}>
            <option value="">All batches and console images</option>
            {batches.map((batch) => <option value={batch.id} key={batch.id}>{batch.name}{batch.archived ? " (archived)" : ""}</option>)}
          </select>
        </Field>
        <div
          className="gallery-categories"
          role="tablist"
          aria-label="Gallery sections"
        >
          {GALLERY_SECTIONS.map(({ id, label, icon: Icon }, index) => (
            <button
              key={id}
              id={"gallery-tab-" + id}
              role="tab"
              aria-selected={gallerySection === id}
              aria-controls="gallery-panel"
              tabIndex={gallerySection === id ? 0 : -1}
              onClick={() => changeGallerySection(id)}
              onKeyDown={(event) => {
                const direction =
                  event.key === "ArrowRight"
                    ? 1
                    : event.key === "ArrowLeft"
                      ? -1
                      : 0;
                const target =
                  event.key === "Home"
                    ? 0
                    : event.key === "End"
                      ? GALLERY_SECTIONS.length - 1
                      : (index + direction + GALLERY_SECTIONS.length) %
                        GALLERY_SECTIONS.length;
                if (direction || event.key === "Home" || event.key === "End") {
                  event.preventDefault();
                  changeGallerySection(GALLERY_SECTIONS[target].id);
                  document
                    .getElementById(
                      "gallery-tab-" + GALLERY_SECTIONS[target].id,
                    )
                    ?.focus();
                }
              }}
            >
              <Icon size={16} />
              {label}
              <span>{galleryCounts[id]}</span>
            </button>
          ))}
        </div>
        <div
          id="gallery-panel"
          role="tabpanel"
          aria-labelledby={"gallery-tab-" + gallerySection}
          aria-busy={galleryLoading}
        >
          <div className="gallery-filters">
            <div className="search-box">
              <Search size={16} />
              <input
                aria-label="Search gallery"
                placeholder={
                  isResources ? "Search resources" : "Search photographs"
                }
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
            {isResources ? (
              <>
                <select
                  aria-label="Resource status"
                  value={resourceFilter}
                  onChange={(event) => setResourceFilter(event.target.value)}
                >
                  {["all", "pending", "approved", "rejected", "used"].map(
                    (value) => (
                      <option key={value} value={value}>
                        {value === "all"
                          ? "All"
                          : value.charAt(0).toUpperCase() + value.slice(1)}{" "}
                        ({resourceCounts[value] || 0})
                      </option>
                    ),
                  )}
                </select>
                <select
                  aria-label="Resource type"
                  value={resourceKind}
                  onChange={(event) => setResourceKind(event.target.value)}
                >
                  <option value="all">All types</option>
                  <option value="identity">Identity</option>
                  <option value="uploaded">Uploads</option>
                  <option value="search">Search</option>
                </select>
              </>
            ) : (
              <>
                <select
                  aria-label="Gallery status"
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                >
                  {(gallerySection === "review"
                    ? [
                        "all",
                        "needs_review",
                        "unchecked",
                        "no_change",
                        "failed",
                        "stopped",
                      ]
                    : [
                        "all",
                        "passed",
                        "saved",
                        "needs_review",
                        "unchecked",
                        "no_change",
                      ]
                  ).map((s) => (
                    <option key={s} value={s}>
                      {s.replaceAll("_", " ")}
                    </option>
                  ))}
                </select>
                {gallerySection !== "enhancements" && (
                  <select
                    aria-label="Gallery source"
                    value={origin}
                    onChange={(e) => setOrigin(e.target.value)}
                  >
                    {(gallerySection === "review"
                      ? [
                          "all",
                          "console",
                          "guided",
                          "auto",
                          "enhancement",
                          "legacy",
                        ]
                      : ["all", "console", "guided", "auto", "legacy"]
                    ).map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))}
                  </select>
                )}
              </>
            )}
          </div>
          {galleryLoading && !loaded ? (
            <div className="empty" role="status">
              <Loader2 className="spin" />
              Loading...
            </div>
          ) : !loaded ? (
            <div className="empty">
              <Images size={32} />
              <h3>
                {isResources
                  ? "No matching resources"
                  : gallerySection === "review"
                    ? "No images awaiting review"
                    : "No images here yet"}
              </h3>
            </div>
          ) : isResources ? (
            <div className="gallery-grid resource-grid">
              {resources.map((resource) => (
                <article className="resource-tile" key={resource.id}>
                  {resource.image ? (
                    <button
                      className="resource-image"
                      onClick={() => {
                        const pictures = resources.filter((row) => row.image);
                        view(
                          pictures.map((row) => row.image!),
                          pictures.findIndex((row) => row.id === resource.id),
                        );
                      }}
                    >
                      <img
                        src={resource.image.thumb}
                        alt={resource.name}
                        loading="lazy"
                      />
                    </button>
                  ) : (
                    <div className="resource-placeholder">
                      <ImagePlus size={24} />
                      <span>Not downloaded</span>
                    </div>
                  )}
                  <div className="resource-caption">
                    <strong>{resource.name}</strong>
                    <div className="resource-badges">
                      <span className={"status " + resource.state}>
                        {resource.state}
                      </span>
                      {resource.used && (
                        <span className="status used">Used</span>
                      )}
                      {resource.committed && !resource.used && (
                        <span className="status">Committed</span>
                      )}
                      <span className="resource-type">{resource.kind}</span>
                    </div>
                    <p>
                      {resource.reason ||
                        resource.check_state.replaceAll("_", " ")}
                    </p>
                    {resource.validation != null && (
                      <details>
                        <summary>Check details</summary>
                        <pre>
                          {JSON.stringify(resource.validation, null, 2)}
                        </pre>
                      </details>
                    )}
                    <div className="button-row">
                      {resource.image && (
                        <a
                          className="icon-button"
                          title="Download resource"
                          aria-label="Download resource"
                          href={resource.image.url + "?download=true"}
                        >
                          <Download size={16} />
                        </a>
                      )}
                      {resource.source_url?.startsWith("https://") && (
                        <a
                          href={resource.source_url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Source
                        </a>
                      )}
                    </div>
                  </div>
                </article>
              ))}
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
                      title="Enhance image"
                      disabled={busy}
                      onClick={() =>
                        setEnhance({
                          image: item.image,
                          item_id: item.id,
                          avatar_id: avatarId,
                        })
                      }
                    >
                      <WandSparkles />
                    </IconButton>
                  </div>
                  <div className="tile-caption">
                    <span>{item.origin}</span>
                    {item.source && (
                      <button
                        className="quiet"
                        onClick={() => {
                          view(
                            gallery.map((row) => row.image),
                            index,
                            item,
                          );
                          setComparing(true);
                        }}
                      >
                        <Columns2 size={15} />
                        Compare
                      </button>
                    )}
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
          {loaded < count && (
            <button
              className="load-more"
              disabled={galleryLoading}
              onClick={() => loadGallery(loaded)}
            >
              {galleryLoading ? "Loading..." : `Load more (${count - loaded})`}
            </button>
          )}
        </div>
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
        <button onClick={() => window.location.reload()}>Reconnect</button>
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
              onClick={() => startChat(null)}
            >
              <Camera size={18} />
              Create
            </button>
            <button
              className={isAvatar ? "selected" : ""}
              onClick={avatarLibrary}
            >
              <Users size={18} />
              Avatar Studio
            </button>
          </nav>
          {scope && avatar && (
            <div className="sidebar-avatar">
              <button className="quiet" onClick={avatarLibrary}>
                <ArrowLeft size={14} />
                All avatars
              </button>
              <div>
                {avatar.face ? (
                  <img src={avatar.face.thumb} alt="" />
                ) : (
                  <UserRound size={26} />
                )}
                <strong>{avatar.name}</strong>
              </div>
            </div>
          )}
          <div className="library-header">
            <span>{scope ? "AVATAR CHATS" : "CHATS"}</span>
          </div>
          <button
            className="new-conversation"
            onClick={() => startChat()}
            disabled={chatLoading}
          >
            <Plus size={16} />
            {scope ? "New avatar chat" : "New chat"}
          </button>
          <div className="search-box compact">
            <Search size={14} />
            <input
              aria-label="Search chats"
              placeholder="Search chats"
              value={chatSearch}
              onChange={(e) => setChatSearch(e.target.value)}
            />
          </div>
          <div className="conversation-list">
            {chats
              .filter((row) =>
                row.title.toLowerCase().includes(chatSearch.toLowerCase()),
              )
              .map((row) => (
                <div
                  key={row.id}
                  className={
                    "conversation-row " + (chatId === row.id ? "selected" : "")
                  }
                >
                  <button
                    className="conversation-select"
                    aria-current={chatId === row.id ? "page" : undefined}
                    onClick={() => selectChat(row.id)}
                    title={row.title}
                  >
                    <MessageSquare size={15} />
                    <span>
                      {row.title}
                      <small>
                        {jobs.some(
                          (job) =>
                            active(job) && job.payload.chat_id === row.id,
                        )
                          ? "Generating..."
                          : row.turn_count
                            ? `${row.turn_count} messages`
                            : "New conversation"}
                      </small>
                    </span>
                  </button>
                  <div className="conversation-actions">
                    <IconButton
                      title={"Rename chat " + row.title}
                      onClick={() => {
                        setRenameChat(row);
                        setChatName(row.title);
                      }}
                    >
                      <Pencil />
                    </IconButton>
                    <IconButton
                      title={"Archive chat " + row.title}
                      onClick={async () => {
                        if (
                          !confirm(
                            "Archive this conversation? Its images and saved history will be kept.",
                          )
                        )
                          return;
                        try {
                          await api("/chats/" + row.id + scopeQuery(scope), {
                            method: "DELETE",
                          });
                          if (chatId === row.id) {
                            const remaining = chats.filter(
                              (chat) => chat.id !== row.id,
                            );
                            setChats(remaining);
                            if (remaining.length)
                              await selectChat(remaining[0].id);
                            else await startChat();
                          } else await refreshConversations();
                        } catch (e) {
                          fail(e);
                        }
                      }}
                    >
                      <Archive />
                    </IconButton>
                  </div>
                </div>
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
                "identity",
                "quality",
                "content",
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
          {route.kind === "notfound" || routeError ? (
            <div className="empty route-error" role="status">
              <h2>Page unavailable</h2>
              <p>
                {routeError || "This address does not match a Studio page."}
              </p>
              <button onClick={() => go("/avatars")}>
                <Users size={16} />
                Avatar Studio
              </button>
              <button onClick={() => go("/chats")}>
                <MessageSquare size={16} />
                Your chats
              </button>
            </div>
          ) : !isAvatar ? (
            chat()
          ) : route.kind === "library" ? (
            <div className="avatar-library">
              <div className="view-toolbar">
                <h2>
                  Your avatars <span className="muted">{avatars.length}</span>
                </h2>
                <button className="primary" onClick={() => setNewAvatar(true)}>
                  <Plus size={16} />
                  New avatar
                </button>
              </div>
              <div className="search-box avatar-search">
                <Search size={16} />
                <input
                  aria-label="Find avatar"
                  placeholder="Find an avatar"
                  value={avatarSearch}
                  onChange={(event) => setAvatarSearch(event.target.value)}
                />
              </div>
              <div className="avatar-grid">
                {avatars
                  .filter((value) =>
                    value.name
                      .toLowerCase()
                      .includes(avatarSearch.toLowerCase()),
                  )
                  .map((value) => {
                    const picture =
                      value.face || value.face_candidate || value.body;
                    return (
                      <button
                        className="avatar-tile"
                        key={value.id}
                        aria-label={"Open avatar " + value.name}
                        onClick={() => openAvatar(value)}
                      >
                        <div className="avatar-portrait">
                          {picture ? (
                            <img
                              src={picture.url}
                              alt={value.name}
                              loading="lazy"
                            />
                          ) : (
                            <UserRound size={48} />
                          )}
                          <span
                            className={
                              "status " + (value.body_locked ? "passed" : "")
                            }
                          >
                            {value.body_locked
                              ? "Ready"
                              : value.face_locked
                                ? "Body setup"
                                : "Face setup"}
                          </span>
                        </div>
                        <div className="avatar-tile-caption">
                          <strong>{value.name}</strong>
                          <ArrowRight size={17} />
                        </div>
                      </button>
                    );
                  })}
                <button
                  className="avatar-tile avatar-new"
                  onClick={() => setNewAvatar(true)}
                >
                  <Plus size={28} />
                  <span>Create avatar</span>
                </button>
              </div>
            </div>
          ) : !avatar || avatar.id !== avatarId ? (
            <div className="empty" role="status">
              <Loader2 className="spin" />
              Loading avatar...
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
          ) : ["identity", "quality", "content"].includes(tab) ? (
            <InfluencerWorkspace
              key={avatar.id + tab}
              avatar={avatar}
              tab={tab as "identity" | "quality" | "content"}
              busy={busy}
              jobs={jobs}
              action={action}
              onError={fail}
              view={view}
              onSaved={refresh}
            />
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
              <div className="app-memory-setting">
                <h3>App-wide memory</h3>
                <label
                  className="check"
                  title="On releases FLUX and vision between stages to reduce memory use. Off keeps them loaded when possible. Manual unload and out-of-memory cleanup remain active."
                >
                  <input
                    type="checkbox"
                    role="switch"
                    aria-label="GPU memory management"
                    checked={config.app_settings.gpu_memory_management}
                    disabled={busy || savingAppSettings}
                    onChange={(e) => {
                      const enabled = e.target.checked;
                      const previous = config.app_settings;
                      appSettingsEpoch.current++;
                      appSettingsPending.current = true;
                      setConfig((current) =>
                        current
                          ? {
                              ...current,
                              app_settings: { gpu_memory_management: enabled },
                            }
                          : current,
                      );
                      setSavingAppSettings(true);
                      api<Config["app_settings"]>("/settings", {
                        method: "PATCH",
                        body: JSON.stringify({
                          gpu_memory_management: enabled,
                        }),
                      })
                        .then((value) =>
                          setConfig((current) =>
                            current
                              ? { ...current, app_settings: value }
                              : current,
                          ),
                        )
                        .catch((error) => {
                          setConfig((current) =>
                            current
                              ? { ...current, app_settings: previous }
                              : current,
                          );
                          fail(error);
                        })
                        .finally(() => {
                          appSettingsEpoch.current++;
                          appSettingsPending.current = false;
                          setSavingAppSettings(false);
                        });
                    }}
                  />
                  GPU memory management
                  <span>
                    {savingAppSettings
                      ? "Saving"
                      : config.app_settings.gpu_memory_management
                        ? "On"
                        : "Off"}
                  </span>
                </label>
                {!config.app_settings.gpu_memory_management && (
                  <p className="memory-warning">
                    Higher memory use; a large workload can run out of RAM or
                    VRAM.
                  </p>
                )}
              </div>
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
                <>
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
                  <Field label="Face reference view">
                    <select
                      value={faceView}
                      onChange={(e) => setFaceView(e.target.value)}
                    >
                      {[
                        "auto",
                        "front",
                        "three-quarter",
                        "profile",
                        "smile",
                      ].map((value) => (
                        <option key={value} value={value}>
                          {value.replaceAll("-", " ")}
                        </option>
                      ))}
                    </select>
                  </Field>
                  <Field label="Body reference view">
                    <select
                      value={bodyView}
                      onChange={(e) => setBodyView(e.target.value)}
                    >
                      {["auto", "body-front", "body-side"].map((value) => (
                        <option key={value} value={value}>
                          {value.replaceAll("-", " ")}
                        </option>
                      ))}
                    </select>
                  </Field>
                </>
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
                      startChat(null);
                      setSettings(false);
                    }}
                  >
                    <Plus size={16} />
                    New conversation
                  </button>
                )}
              </div>
            </div>
          </Modal>
        )}
        {renameChat && (
          <Modal
            title="Rename conversation"
            onClose={() => setRenameChat(null)}
          >
            <div className="settings-content">
              <Field label="Conversation name">
                <input
                  autoFocus
                  value={chatName}
                  maxLength={100}
                  onChange={(event) => setChatName(event.target.value)}
                />
              </Field>
              <button
                className="primary"
                disabled={!chatName.trim()}
                onClick={async () => {
                  try {
                    await write(
                      "/chats/" + renameChat.id + scopeQuery(scope),
                      { name: chatName },
                      "PATCH",
                    );
                    setRenameChat(null);
                    await refreshConversations();
                  } catch (e) {
                    fail(e);
                  }
                }}
              >
                <Check size={16} />
                Save name
              </button>
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
                {comparing && viewerSource ? (
                  <div className="comparison">
                    <figure>
                      <img src={comparisonUrl(viewerSource)} alt="Original" />
                      <figcaption>Original</figcaption>
                    </figure>
                    <figure>
                      <img
                        src={comparisonUrl(viewer.images[viewer.index])}
                        alt="Reconstruction"
                      />
                      <figcaption>Enhanced</figcaption>
                    </figure>
                  </div>
                ) : (
                  <img
                    src={viewer.images[viewer.index].url}
                    width={viewer.images[viewer.index].width}
                    height={viewer.images[viewer.index].height}
                    alt="Full-size photograph"
                  />
                )}
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
                  {viewerSource && (
                    <>
                      <button
                        aria-pressed={comparing}
                        onClick={() => setComparing((value) => !value)}
                      >
                        <Columns2 size={16} />
                        Compare
                      </button>
                      {comparing && (
                        <Field label="Comparison area">
                          <select
                            value={comparisonRegion}
                            onChange={(e) =>
                              setComparisonRegion(e.target.value)
                            }
                          >
                            <option value="full">Full photograph</option>
                            {detailRegions.map((region, index) => (
                              <option key={index} value={String(index)}>
                                {region.label}
                              </option>
                            ))}
                          </select>
                        </Field>
                      )}
                    </>
                  )}
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
                  {(viewer.item || viewer.chat_id) && (
                    <button
                      disabled={busy}
                      onClick={() => {
                        setEnhance({
                          image: viewer.images[viewer.index],
                          avatar_id: viewer.avatar_id || null,
                          chat_id: viewer.chat_id,
                          item_id: viewer.item?.id,
                        });
                        setViewer(null);
                      }}
                    >
                      <WandSparkles size={16} />
                      Enhance
                    </button>
                  )}
                  {viewer.item && (
                    <>
                      <button
                        disabled={busy}
                        onClick={() =>
                          action(
                            {
                              kind: "quality_check",
                              avatar_id: viewer.avatar_id,
                              item_id: viewer.item!.id,
                            },
                            () => setViewer(null),
                          )
                        }
                      >
                        <ScanFace size={16} />
                        Check quality
                      </button>
                      {[
                        "needs_review",
                        "unchecked",
                        "no_change",
                        "failed",
                        "stopped",
                      ].includes(viewer.item.status) && (
                        <button
                          disabled={busy}
                          onClick={async () => {
                            const id = viewer.item!.id;
                            try {
                              const approved = await write<GalleryItem>(
                                `/avatars/${viewer.avatar_id}/gallery/${id}/approve`,
                              );
                              setViewer((old) =>
                                old?.item?.id === id
                                  ? { ...old, item: approved }
                                  : old,
                              );
                              await loadGallery();
                            } catch (error) {
                              fail(error);
                            }
                          }}
                        >
                          <Check size={16} />
                          Approve
                        </button>
                      )}
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
              {viewerPlan && (
                <details className="image-details enhancement-analysis">
                  <summary>Enhancement analysis</summary>
                  <p>
                    {viewerPlan.selection === "manual_override" ? "Source analysis: skipped (selected profile)" : !viewerPlan.automatic ? "Custom enhancement prompt" : viewerPlan.analysis.mock_not_quality_evidence
                      ? "Local preview simulation"
                      : `Source analysis: ${viewerPlan.analysis.status}${viewerPlan.analysis.cache_hit ? " (reused)" : ""} (unverified estimate)`}
                  </p>
                  {viewerPlan.automatic && (
                    <dl>
                      {(viewerPlan.analysis.profile_id
                        ? ["profile_name", "medium", "scene", "people_role", "people_count", "framing", "presentation", "focus", "condition"]
                        : ["scene", "materials", "lighting", "condition", "skin_texture", "skin_marks"]
                      ).map((key) => (
                        <div key={key}>
                          <dt>
                            {key === "profile_name" ? "Prompt profile" : key.replaceAll("_", " ")}
                          </dt>
                          <dd>
                            {(key === "profile_name" && viewerPlan.profile_name
                              ? viewerPlan.profile_name
                              : Array.isArray(viewerPlan.analysis[key])
                              ? (viewerPlan.analysis[key] as string[]).join(
                                  ", ",
                                ) || "uncertain"
                              : String(viewerPlan.analysis[key] || "uncertain")
                            ).replaceAll("_", " ")}
                          </dd>
                        </div>
                      ))}
                    </dl>
                  )}
                  {!viewerPlan.automatic && viewerPlan.profile_name && <p>{viewerPlan.profile_name}</p>}
                  {viewerPlan.focus_rules?.length ? (
                    <p>
                      Focus rules: {viewerPlan.focus_rules.map((rule) => rule.replaceAll("_", " ")).join(", ")}
                    </p>
                  ) : null}
                  {viewerPlan.selection === "manual_override" && <p>Profile selection: manual override</p>}
                  {Array.isArray(viewerPlan.analysis.routing_notes) && viewerPlan.analysis.routing_notes.map((note) => (
                    <p key={String(note)}>{String(note)}</p>
                  ))}
                  {textureReview && (
                    <p>
                      {viewerDiagnostics?.content_review ? "Content review: " : "Texture review: "}
                      {textureReview.status.replaceAll("_", " ")}
                    </p>
                  )}
                  {textureReview?.reasons?.map((reason) => (
                    <p className="memory-warning" key={reason}>
                      {reason}
                    </p>
                  ))}
                  {enhancementReview && (
                    <p>
                      Detail review:{" "}
                      {enhancementReview.status.replaceAll("_", " ")}
                    </p>
                  )}
                  {enhancementReview?.reasons?.map((reason) => (
                    <p className="memory-warning" key={reason}>
                      {reason}
                    </p>
                  ))}
                  {Boolean(viewerPlan.analysis.answers) && (
                    <details>
                      <summary>Analysis questions and answers</summary>
                      <pre>
                        {JSON.stringify(viewerPlan.analysis.answers, null, 2)}
                      </pre>
                    </details>
                  )}
                  <details>
                    <summary>Applied reconstruction prompt</summary>
                    <p>{viewerPlan.prompt}</p>
                  </details>
                  {Boolean(viewerDiagnostics?.sampling) && <details>
                    <summary>Sampling details</summary>
                    <pre>{JSON.stringify(viewerDiagnostics?.sampling, null, 2)}</pre>
                  </details>}
                </details>
              )}
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
                        manual_review: viewer.item.manual_review,
                        identity_version: (
                          viewer.item as GalleryItem & {
                            identity_version?: string;
                          }
                        ).identity_version,
                        recipe: (
                          viewer.item as GalleryItem & { recipe?: unknown }
                        ).recipe,
                      },
                      null,
                      2,
                    )}
                  </pre>
                </details>
              )}
              {comparing && detailEvidence && (
                <p className="comparison-evidence muted">
                  {detailEvidence.advisory?.replaceAll("_", " ")} · Detail
                  energy {detailEvidence.source_detail_energy} →{" "}
                  {detailEvidence.result_detail_energy}
                </p>
              )}
            </div>
          </Modal>
        )}
        {enhance && (
          <Modal
            title="Enhance image"
            className="enhancement-modal"
            onClose={() => setEnhance(null)}
          >
            <div className="settings-content enhancement-layout">
              <img
                className="enhance-preview"
                src={enhance.image.thumb}
                alt="Source photograph"
              />
              <div className="enhancement-controls">
                <Field label="Sampling">
                  <select value={enhanceSampling} onChange={(e) => setEnhanceSampling(e.target.value)}>
                    <option value="reference">FLUX.2 reference reconstruction</option>
                    <option value="img2img">Low-noise img2img (experimental)</option>
                  </select>
                </Field>
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
                {enhanceSampling === "img2img" && <Range
                  label="Reconstruction strength"
                  value={strength}
                  min={0.35}
                  max={1}
                  step={0.05}
                  onChange={setStrength}
                />}
                <Field label="Prompt profile">
                  <select value={enhanceProfile} onChange={(e) => setEnhanceProfile(e.target.value)}>
                    <option value="auto">Automatic</option>
                    {config.enhancement_profiles?.map((profile) => <option key={profile.id} value={profile.id}>{profile.name}</option>)}
                    <option value="custom">Custom</option>
                  </select>
                </Field>
              </div>
              <div className="enhancement-prompt">
                <label className="check-field">
                  <input
                    type="checkbox"
                    checked={preserveTone}
                    onChange={(e) => setPreserveTone(e.target.checked)}
                  />
                  Preserve source lighting and colors
                </label>
                {enhanceProfile === "auto" && <p role="status">Automatic profile: pending source analysis</p>}
                {selectedEnhancementProfile && <details>
                  <summary>Selected profile prompt</summary>
                  <p>{selectedEnhancementProfile.prompt}</p>
                  <button onClick={() => { setEnhancementPrompt(selectedEnhancementProfile.prompt + (enhanceInstructions.trim() ? "\n\nAdditional reconstruction instructions: " + enhanceInstructions.trim() : "")); setEnhanceProfile("custom"); }}>
                    <Copy size={16} /> Use as custom prompt
                  </button>
                </details>}
                <Field label={customEnhancement ? "Custom enhancement prompt" : "Additional instructions (optional)"}>
                  <textarea
                    rows={8}
                    maxLength={16000}
                    value={customEnhancement ? reconstructionPrompt : enhanceInstructions}
                    onChange={(e) => customEnhancement ? setEnhancementPrompt(e.target.value) : setEnhanceInstructions(e.target.value)}
                  />
                </Field>
              </div>
              <div className="enhancement-actions">
                <button
                  onClick={() => customEnhancement ? setEnhancementPrompt("") : setEnhanceInstructions("")}
                  disabled={!(customEnhancement ? enhancementPrompt : enhanceInstructions)}
                >
                  <Undo2 size={16} />
                  Clear text
                </button>
                <button
                  className="primary"
                  disabled={busy || (customEnhancement && !reconstructionPrompt.trim())}
                  onClick={() =>
                    action(
                      {
                        kind: "enhance",
                        avatar_id: enhance.avatar_id,
                        chat_id: enhance.chat_id,
                        item_id: enhance.item_id,
                        image: enhance.image.path,
                        method: enhanceMethod,
                        sampling_mode: enhanceSampling,
                        enhancement_profile: customEnhancement ? "auto" : enhanceProfile,
                        scale,
                        strength,
                        reconstruction_prompt: customEnhancement ? reconstructionPrompt : config.reconstruction_prompt,
                        additional_instructions: customEnhancement ? "" : enhanceInstructions,
                        adaptive_prompt: !customEnhancement,
                        preserve_tone: preserveTone,
                        steps: 4,
                      },
                      () => setEnhance(null),
                    )
                  }
                >
                  <WandSparkles size={16} />
                  Enhance image
                </button>
              </div>
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
        {batchManagerOpen && <Modal title="Gallery batches" wide onClose={() => setBatchManagerOpen(false)}>
          <div className="settings-content">
            <button className="primary" disabled={busy || !avatar?.body_locked} onClick={() => {setBatchManagerOpen(false); setBatchName(""); setBatchOpen(true);}}><Plus size={16} /> Create batch</button>
            <GalleryBatches batches={batches} busy={busy} open={(id) => {selectBatch(id); setBatchManagerOpen(false);}}
              resume={resumeBatch} manage={async (id, operation, name) => {
                try {
                  await write("/avatars/" + avatarId + "/gallery/batches/" + id, {action: operation, name: name || ""});
                  await loadBatches(); await loadFindings();
                } catch (e) { fail(e); }
              }} />
          </div>
        </Modal>}
        {batchOpen && (
          <Modal title="Gallery batch" onClose={() => setBatchOpen(false)}>
            <div className="settings-content">
              <Field label="Batch name"><input value={batchName} onChange={(e) => setBatchName(e.target.value)} maxLength={120} /></Field>
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
                onClick={async () => {
                  const job = await action(
                    {
                      kind: "gallery",
                      avatar_id: avatarId,
                      new_batch: true,
                      batch_name: batchName,
                      quantity: batchQuantity,
                      aspect,
                      theme: batchTheme,
                      require_face: requireFace,
                    },
                  );
                  if (job) {
                      selectBatch(String(job.payload.batch_id));
                      setBatchOpen(false);
                      setFindingsOpen(true);
                  }
                }}
              >
                <ArrowRight size={16} />
                Start batch
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
                  {findings?.job?.name && <strong>{findings.job.name}: </strong>}
                  {findings?.job?.message ||
                    findings?.job?.phase ||
                    "No active batch"}
                </span>
                <div className="button-row">
                  <IconButton title="Refresh pipeline" onClick={loadFindings}>
                    <RefreshCw />
                  </IconButton>
                  <button
                    disabled={busy || !findings?.job || findings.job.archived || ["complete", "exhausted"].includes(findings.job.status)}
                    onClick={() => resumeBatch(findings.job.id)}
                  >
                    <ArrowRight size={15} />
                    Resume batch
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
createRoot(document.getElementById("root")!).render(
  <BrowserRouter>
    <App />
  </BrowserRouter>,
);
