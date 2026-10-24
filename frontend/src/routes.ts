import { generatePath, matchRoutes } from "react-router-dom";

export type AvatarTab =
  | "guide"
  | "face"
  | "body"
  | "console"
  | "gallery"
  | "identity"
  | "quality"
  | "content";
export type GallerySection =
  "generations" | "enhancements" | "resources" | "review";
export type StudioRoute = {
  kind: "main" | "library" | "avatar" | "notfound";
  avatarId?: string;
  chatId?: string;
  tab?: AvatarTab;
  gallerySection?: GallerySection;
  draft?: boolean;
};

const definitions = [
  { path: "/", handle: { kind: "main", draft: true } },
  { path: "/chats", handle: { kind: "main", draft: true } },
  { path: "/chats/new", handle: { kind: "main", draft: true } },
  { path: "/chats/:chatId", handle: { kind: "main" } },
  { path: "/avatars", handle: { kind: "library" } },
  { path: "/avatars/:avatarId", handle: { kind: "avatar" } },
  {
    path: "/avatars/:avatarId/console/new",
    handle: { kind: "avatar", tab: "console", draft: true },
  },
  ...(
    [
      "guide",
      "face",
      "body",
      "console",
      "gallery",
      "identity",
      "quality",
      "content",
    ] as const
  ).map((tab) => ({
    path: "/avatars/:avatarId/" + tab,
    handle: { kind: "avatar", tab },
  })),
  {
    path: "/avatars/:avatarId/console/:chatId",
    handle: { kind: "avatar", tab: "console" },
  },
  ...(["generations", "enhancements", "resources", "review"] as const).map(
    (gallerySection) => ({
      path:
        "/avatars/:avatarId/gallery/" +
        (gallerySection === "review" ? "needs-review" : gallerySection),
      handle: { kind: "avatar", tab: "gallery", gallerySection },
    }),
  ),
];

export function studioRoute(pathname: string): StudioRoute {
  const match = matchRoutes(definitions, pathname)?.[0];
  if (
    !match ||
    Object.values(match.params).some(
      (value) => !value || !/^[a-zA-Z0-9_-]{1,100}$/.test(value),
    )
  ) {
    return { kind: "notfound" };
  }
  return { ...match.route.handle, ...match.params } as StudioRoute;
}

export function chatPath(chatId: string, avatarId: string | null = null) {
  return avatarId
    ? generatePath("/avatars/:avatarId/console/:chatId", { avatarId, chatId })
    : generatePath("/chats/:chatId", { chatId });
}
export function newChatPath(avatarId: string | null = null) {
  return avatarId ? avatarPath(avatarId, "console") + "/new" : "/chats/new";
}

export function avatarPath(
  avatarId: string,
  tab?: AvatarTab,
  section: GallerySection = "generations",
) {
  const base = generatePath("/avatars/:avatarId", { avatarId });
  if (!tab) return base;
  return tab === "gallery"
    ? base + "/gallery/" + (section === "review" ? "needs-review" : section)
    : base + "/" + tab;
}
