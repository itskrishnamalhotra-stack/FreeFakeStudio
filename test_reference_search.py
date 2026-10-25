"""Search-provider contract and real icrawler parsing without network or AI calls."""
import importlib.util
from types import SimpleNamespace
import unittest
from unittest import mock

import avatar_gallery as gallery
import reference_search as search


def page(html, url="https://www.google.com/search?q=fixture&tbm=isch"):
    return SimpleNamespace(content=html.encode(), url=url)


@unittest.skipUnless(importlib.util.find_spec("icrawler"), "Install icrawler==0.6.10 for the optional Google parser tests")
class GoogleParserTests(unittest.TestCase):
    def test_originals_keep_source_attribution_and_never_use_search_query_as_evidence(self):
        response = page('''<a href="/imgres?imgurl=https%3A%2F%2Fcdn.example.test%2Foriginal.webp&amp;imgrefurl=https%3A%2F%2Fexample.test%2Ffashion">
          <img alt="Adult swimwear editorial" src="https://encrypted-tbn0.gstatic.com/preview"></a>
          <script>let originals = [["https://cdn.example.test/second.jpg",1000,800]];</script>''')
        rows = search.parse_google_response(response)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source_url"], "https://example.test/fashion")
        self.assertEqual(rows[0]["title"], "Adult swimwear editorial")
        self.assertEqual(rows[1]["image_url"], "https://cdn.example.test/second.jpg")
        self.assertEqual(rows[1]["title"], "")
        self.assertEqual(rows[1]["description"], "")
        self.assertEqual({row["search_provider"] for row in rows}, {"google"})

    def test_empty_javascript_captcha_and_consent_responses_are_not_success(self):
        for response, message in ((page("<p>No image results</p>"), "no usable"),
                                  (page("<p>Please enable JavaScript</p>"), "JavaScript"),
                                  (page("<p>Update your browser</p>"), "JavaScript"),
                                  (page("<p>Unusual traffic</p>"), "blocked"),
                                  (page("<form></form>", "https://consent.google.com/"), "blocked")):
            with self.subTest(message=message), self.assertRaisesRegex(search.ImageSearchError, message):
                search.parse_google_response(response)

    def test_thumbnail_and_duplicate_urls_are_not_returned_as_distinct_originals(self):
        rows = search.parse_google_response(page('''<script>let urls = [["https://encrypted-tbn0.gstatic.com/thumb.jpg"],
            ["https://cdn.example.test/a.jpg"],["https://cdn.example.test/a.jpg"]];</script>'''))
        self.assertEqual([row["image_url"] for row in rows], ["https://cdn.example.test/a.jpg"])


class SearchProviderTests(unittest.TestCase):
    def test_old_requests_default_to_tavily_and_invalid_source_is_rejected(self):
        self.assertEqual(search.normalize_provider(None), "tavily")
        with self.assertRaises(ValueError):
            search.normalize_provider("unknown")
        with mock.patch.object(search, "google_images") as google, \
             mock.patch.object(gallery, "_tavily_search", return_value={"images": ["https://example.test/a.jpg"]}) as tavily:
            rows = search.search("exact description")
        google.assert_not_called()
        tavily.assert_called_once_with("exact description", max_results=20)
        self.assertEqual(rows[0]["search_provider"], "tavily")

    def test_google_success_does_not_call_or_require_tavily(self):
        attempts = []
        with mock.patch.object(search, "google_images", return_value=[{"image_url": "https://example.test/a.jpg", "search_provider": "google"}]) as google, \
             mock.patch.object(gallery, "_tavily_search") as tavily, \
             mock.patch.dict(gallery.os.environ, {"FFS_AVATAR_REFERENCE_DOMAINS": "instagram.com"}):
            rows = search.search("exact description", "google_fallback", offset=100, on_attempt=attempts.append)
        google.assert_called_once_with("exact description", offset=100)
        tavily.assert_not_called()
        self.assertEqual(attempts[0]["provider"], "google")
        self.assertEqual(attempts[0]["status"], "success")
        self.assertEqual(rows[0]["search_query"], "exact description")

    def test_google_only_failure_is_not_silently_replaced(self):
        attempts = []
        with mock.patch.object(search, "google_images", side_effect=search.ImageSearchError("Google blocked")), \
             mock.patch.object(gallery, "_tavily_search") as tavily:
            with self.assertRaisesRegex(search.ImageSearchError, "blocked"):
                search.search("query", "google", on_attempt=attempts.append)
        tavily.assert_not_called()
        self.assertEqual(attempts[0]["status"], "failed")

    def test_explicit_fallback_reports_both_sources_and_labels_actual_result_origin(self):
        attempts = []
        with mock.patch.object(search, "google_images", side_effect=search.ImageSearchError("JavaScript page")), \
             mock.patch.object(gallery, "_tavily_search", return_value={"images": ["https://example.test/a.jpg"]}):
            rows = search.search("query", "google_fallback", on_attempt=attempts.append)
        self.assertEqual([(row["provider"], row["status"]) for row in attempts], [("google", "failed"), ("tavily", "success")])
        self.assertEqual(rows[0]["search_provider"], "tavily")
        self.assertIn("JavaScript", attempts[0]["message"])

    def test_cancelled_google_search_does_not_trigger_fallback(self):
        with mock.patch.object(search, "google_images", side_effect=gallery.GalleryCancelled("Stopped")), \
             mock.patch.object(gallery, "_tavily_search") as tavily:
            with self.assertRaises(gallery.GalleryCancelled):
                search.search("query", "google_fallback")
        tavily.assert_not_called()

    def test_failed_fallback_remains_a_failure(self):
        attempts = []
        with mock.patch.object(search, "google_images", side_effect=search.ImageSearchError("Google blocked")), \
             mock.patch.object(gallery, "_tavily_search", side_effect=gallery.GalleryAPIError("Tavily", 503, "Busy", True)):
            with self.assertRaises(gallery.GalleryAPIError):
                search.search("query", "google_fallback", on_attempt=attempts.append)
        self.assertEqual([row["status"] for row in attempts], ["failed", "failed"])

    def test_google_http_failure_and_size_limits_close_response(self):
        response = mock.Mock(status_code=429)
        with mock.patch.object(search.requests, "get", return_value=response):
            with self.assertRaisesRegex(search.ImageSearchError, "HTTP 429"):
                search.google_images("query")
        response.close.assert_called_once()
        response = mock.Mock(status_code=200, url="https://www.google.com/search")
        response.iter_content.return_value = [b"x" * (search.MAX_SEARCH_BYTES + 1)]
        with mock.patch.object(search.requests, "get", return_value=response):
            with self.assertRaisesRegex(search.ImageSearchError, "bounded"):
                search.google_images("query")
        response.close.assert_called_once()

    def test_google_request_does_not_inherit_the_tavily_instagram_filter(self):
        response = mock.Mock(status_code=200, url="https://www.google.com/search")
        response.iter_content.return_value = [b"<html>fixture</html>"]
        with mock.patch.dict(gallery.os.environ, {"FFS_AVATAR_REFERENCE_DOMAINS": "instagram.com"}), \
             mock.patch.object(search.requests, "get", return_value=response) as get, \
             mock.patch.object(search, "parse_google_response", return_value=[{"image_url": "https://example.test/a.jpg"}]):
            search.google_images("exact description", offset=100)
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["q"], "exact description")
        self.assertEqual(params["start"], 100)
        self.assertNotIn("site:", params["q"])


if __name__ == "__main__":
    unittest.main()
