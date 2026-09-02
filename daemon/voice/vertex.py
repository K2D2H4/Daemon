"""Vertex AI as a second transport for the same Live API.

The protocol body is identical to the API-key endpoint's - the same
proto-over-JSON `setup`, the same `serverContent`, the same tool frames - so
`GeminiLiveSession` needs none of this. Exactly three things differ, and they are
all this module supplies: the URI, the auth header, and how the model is named.

Why it is worth a second transport (measured 2026-09-02, Seoul -> us-central1,
one 3.56 s Korean utterance streamed at 1x realtime, 5 trials per arm,
docs/design/vertex-live-transport.md):

    Vertex   gemini-live-2.5-flash-native-audio            1430 ms  (41 ms spread)
    API key  gemini-3.1-flash-live-preview                 1723 ms
    API key  gemini-2.5-flash-native-audio-preview-12-2025 3137 ms  (772 ms spread)

`gemini-live-2.5-flash-native-audio` exists **only here**: the API-key endpoint
closes 1008 "is not found" for it and never lists it. The same measurement also
found the reverse - Vertex has no conversational live model newer than 2.5, while
the newer generation (3.1 live) exists only on the API-key endpoint - which is why
both transports stay, rather than one replacing the other.

Nothing outside `daemon/app.py` imports this module, the same rule the session
classes follow.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from daemon.voice.gemini_live import GeminiLiveError

logger = logging.getLogger(__name__)

SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# Which models and regions this endpoint serves are in `daemon/config.py`
# (VERTEX_LIVE_MODELS, VERTEX_LIVE_LOCATIONS): the admin's option lists need them
# and config is foundation, so they live where a caller can read them without
# importing the voice layer - and PortAudio with it.

API_VERSION = "v1beta1"
"""Vertex's own version, not the API-key endpoint's `v1beta`, and a different
service name with it. Both were read off the google-genai SDK rather than guessed
(`google/genai/live.py`), then confirmed against a live session."""


def ws_url(location: str) -> str:
    """The regional Live endpoint. A wrong region here is a handshake 404, not a
    slow session, so the caller's location is not defaulted quietly."""
    if not location:
        raise ValueError("a Vertex location is required (DAEMON_VERTEX_LOCATION)")
    return (
        f"wss://{location}-aiplatform.googleapis.com/ws/"
        f"google.cloud.aiplatform.{API_VERSION}.LlmBidiService/BidiGenerateContent"
    )


def model_path(project: str, location: str, model: str) -> str:
    """`projects/../locations/../publishers/google/models/..`, which is what Vertex
    wants in `setup.model` - the API-key endpoint's bare `models/{id}` is rejected.

    An id that already carries a full path is passed through, so a value copied
    from the Vertex console works as typed.
    """
    if not model:
        raise ValueError("a model id is required (DAEMON_GEMINI_LIVE_MODEL)")
    if model.startswith("projects/"):
        return model
    if not project:
        raise ValueError("a Vertex project is required (DAEMON_VERTEX_PROJECT)")
    if model.startswith("publishers/"):
        return f"projects/{project}/locations/{location}/{model}"
    return f"projects/{project}/locations/{location}/publishers/google/models/{model}"


def auth_headers(credentials_path: str = "") -> Callable[[], dict[str, str]]:
    """A provider of `Authorization: Bearer ..`, refreshed as it expires.

    A callable rather than a header dict on purpose: an access token lives about
    an hour, a session reconnects for its own reasons, and a reconnect that reuses
    the token it was handed at startup dies at the handshake once that hour is up.
    `GeminiLiveSession` calls this per connect attempt.

    Blocking - it does network I/O on refresh - so the session calls it in a
    thread. Credential failures are raised permanent: every one of them (no
    credentials, expired ADC, a key file that is not there) needs a person, and
    the alternative is a daemon that retries forever while sounding healthy.
    """
    lock = threading.Lock()
    state: dict[str, object] = {}

    def provider() -> dict[str, str]:
        with lock:
            credentials = state.get("credentials")
            if credentials is None:
                credentials = _load_credentials(credentials_path)
                state["credentials"] = credentials
            if not (getattr(credentials, "token", None) and credentials.valid):
                _refresh(credentials)
            token = getattr(credentials, "token", "")
            if not token:
                raise GeminiLiveError(
                    "Vertex credentials produced no access token", permanent=True
                )
            return {"Authorization": f"Bearer {token}"}

    return provider


def _load_credentials(credentials_path: str) -> object:
    """Service-account file if one is named, Application Default Credentials
    otherwise. Imported here rather than at module import: `google-auth` ships in
    the `voice` extra and an install on the API-key transport never needs it."""
    try:
        import google.auth
        from google.oauth2 import service_account
    except ImportError as exc:
        raise GeminiLiveError(
            "the Vertex transport needs the google-auth package, which ships in this "
            "project's `voice` extra - reinstall with that extra, or set "
            "DAEMON_GEMINI_LIVE_TRANSPORT=api_key to stay on the endpoint that needs "
            f"only GEMINI_API_KEY: {exc}",
            permanent=True,
        ) from None

    try:
        if credentials_path:
            return service_account.Credentials.from_service_account_file(
                credentials_path, scopes=[SCOPE]
            )
        credentials, project = google.auth.default(scopes=[SCOPE])
        logger.info("vertex: using Application Default Credentials (project %s)", project)
        return credentials
    except Exception as exc:
        raise GeminiLiveError(
            f"Vertex credentials could not be loaded: {type(exc).__name__}: {exc}. "
            "Point GOOGLE_APPLICATION_CREDENTIALS at a service-account key, or run "
            "`gcloud auth application-default login`.",
            permanent=True,
        ) from None


def _refresh(credentials: object) -> None:
    try:
        from google.auth.transport.requests import Request

        credentials.refresh(Request())  # type: ignore[attr-defined]
    except Exception as exc:
        raise GeminiLiveError(
            f"Vertex credentials could not be refreshed: {type(exc).__name__}: {exc}. "
            "An expired Application Default login is the usual cause - "
            "`gcloud auth application-default login` re-authenticates it.",
            permanent=True,
        ) from None
