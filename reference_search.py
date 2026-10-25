"""Optional Google Images discovery; downloads and approval stay in the gallery pipeline."""
from urllib.parse import parse_qs, urljoin, urlsplit
from types import SimpleNamespace

import requests

PROVIDERS = ("tavily", "google", "google_fallback")
MAX_SEARCH_BYTES = 4 * 1024 * 1024


class ImageSearchError(RuntimeError):
    pass


def normalize_provider(value=None):
    # A missing value must preserve the behavior of batches made before this feature.
    value = value or "tavily"
    if value not in PROVIDERS:
        raise ValueError("Choose Tavily, Google Images, or Google then Tavily")
    return value


def _original_url(value):
    try:
        if not isinstance(value, str) or any(character.isspace() or character in '<>"\\' for character in value):
            return None
        parsed = urlsplit(str(value))
        host = (parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
            return None
        if host.endswith(("gstatic.com", "googleusercontent.com")) or host in {"google.com", "www.google.com"}:
            return None
        return str(value)
    except ValueError:
        return None


def parse_google_response(response, limit=100):
    try:
        from bs4 import BeautifulSoup
        from icrawler.builtin import GoogleParser
    except ImportError as exc:
        raise ImageSearchError("Google Images requires icrawler==0.6.10. Update and rerun notebook setup; Tavily remains available.") from exc
    soup = BeautifulSoup(response.content, "lxml")
    text = soup.get_text(" ", strip=True).lower()
    host = urlsplit(response.url).hostname or ""
    if ("consent.google" in host or "/sorry/" in response.url or soup.select_one('form[action*="/sorry"]')
            or any(term in text for term in ("unusual traffic", "verify you're not a robot", "verify you are not a robot"))):
        raise ImageSearchError("Google Images blocked this runtime with a consent or CAPTCHA page. No challenge was bypassed. Use Tavily or try Google later.")
    rows = {}

    def add(url, title="", source_url=""):
        url = _original_url(url)
        if not url:
            return
        row = rows.setdefault(url, {"image_url": url, "title": "", "description": "", "source_url": "",
                                    "source_type": "google_image", "search_provider": "google"})
        if title and not row["title"]:
            row["title"] = str(title)[:1000]
        if _original_url(source_url) and not row["source_url"]:
            row["source_url"] = source_url

    # Preserve source attribution when Google supplies normal image-result links.
    for anchor in soup.select('a[href]'):
        href = urljoin("https://www.google.com", anchor.get("href", ""))
        parsed = urlsplit(href)
        if parsed.hostname not in {"google.com", "www.google.com"} or parsed.path != "/imgres":
            continue
        values = parse_qs(parsed.query)
        image = anchor.find("img")
        title = (image.get("alt", "") if image else "") or anchor.get_text(" ", strip=True)
        add(values.get("imgurl", [""])[0], title, values.get("imgrefurl", [""])[0])
    # Use the library's Google parser for script-embedded originals. It performs
    # discovery only: our existing downloader owns limits, file checks and persistence.
    try:
        parser = GoogleParser(1, None, None)
        tasks = parser.parse(response) or []
        for task in tasks:
            if isinstance(task, dict):
                add(task.get("file_url", ""), task.get("title", ""), task.get("source_url", ""))
    except (TypeError, ValueError, UnicodeError, AttributeError) as exc:
        if not rows:
            raise ImageSearchError("Google Images returned an unsupported result format; no usable original URLs were parsed. Tavily remains available.") from exc
    if not rows:
        if any(term in text for term in ("enable javascript", "update your browser", "browser is not supported")) or soup.select_one('a[href*="/httpservice/retry/enablejs"]'):
            raise ImageSearchError("Google Images requires JavaScript/browser verification on this runtime. The lightweight crawler cannot read these results. Use Tavily or Google then Tavily.")
        raise ImageSearchError("Google Images returned no usable original image URLs. This may be an empty search or an unsupported Google page; no images were approved.")
    return list(rows.values())[:max(1, min(100, int(limit)))]


def google_images(query, offset=0, limit=100):
    import avatar_gallery as gallery
    gallery.check_cancelled()
    response = None
    try:
        response = requests.get("https://www.google.com/search", params={
            "q": query, "tbm": "isch", "hl": "en", "start": offset, "ijn": offset // 100,
            "tbs": "itp:photo,isz:l"}, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36"},
            timeout=(10, 25), stream=True)
        if response.status_code != 200:
            raise ImageSearchError(f"Google Images returned HTTP {response.status_code}. Use Tavily or retry later.")
        data = bytearray()
        for chunk in response.iter_content(128 * 1024):
            gallery.check_cancelled()
            data.extend(chunk)
            if len(data) > MAX_SEARCH_BYTES:
                raise ImageSearchError("Google Images response exceeded the bounded search-page size")
        return parse_google_response(SimpleNamespace(content=bytes(data), url=response.url), limit)
    except requests.RequestException as exc:
        # Do not expose request headers, proxy credentials or complete exception URLs.
        raise ImageSearchError(f"Google Images connection failed ({type(exc).__name__}); saved resources are preserved") from exc
    finally:
        if response is not None:
            response.close()


def search(query, provider="tavily", offset=0, on_attempt=None):
    import avatar_gallery as gallery
    provider = normalize_provider(provider)

    def report(source, status, count=0, message=""):
        if on_attempt:
            on_attempt({"provider": source, "query": query, "offset": offset,
                        "status": status, "found": count, "message": message})

    if provider in {"google", "google_fallback"}:
        try:
            rows = google_images(query, offset=offset)
        except ImageSearchError as exc:
            report("google", "failed", message=str(exc))
            if provider == "google":
                raise
            gallery.check_cancelled()
        else:
            report("google", "success", len(rows))
            return [{**row, "search_query": query} for row in rows]
    try:
        rows = gallery._candidate_images(gallery._tavily_search(query, max_results=20))
    except Exception as exc:
        report("tavily", "failed", message=type(exc).__name__)
        raise
    report("tavily", "success", len(rows))
    return [{**row, "search_provider": "tavily", "search_query": query} for row in rows]
