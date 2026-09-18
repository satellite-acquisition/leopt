"""Fetch TLEs from CelesTrak or Space-Track and poll for newly catalogued objects.

CelesTrak queries need no credentials; Space-Track uses an authenticated
session. Fetch functions are injectable for offline tests. The returned TLE
lines can be passed to the orbit parser and particle estimator.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from http.cookiejar import CookieJar

# A pulled element set: the two TLE lines plus an optional name/norad tag.
TlePair = tuple[str, str]


def extract_tle_pair(text: str) -> TlePair | None:
    """Pull the first '1 '/'2 ' element-line pair out of a catalog response.

    Returns None (rather than raising) when the response has no element lines,
    which is the normal "not catalogued yet" case a poller must handle.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    l1 = next((ln for ln in lines if ln.startswith("1 ") and len(ln) >= 69), None)
    l2 = next((ln for ln in lines if ln.startswith("2 ") and len(ln) >= 69), None)
    if l1 and l2:
        return l1, l2
    return None


def _http_get(url: str, timeout: float, opener: urllib.request.OpenerDirector | None = None) -> str:
    """GET `url` and return the decoded body (stdlib urllib).

    Honours `LEOPT_OFFLINE`: on an air-gapped appliance an accidental outbound
    catalog fetch must fail *fast* (never hang mid-pass), so we short-circuit
    before touching the socket.
    """
    if os.environ.get("LEOPT_OFFLINE"):
        raise OSError("offline mode (LEOPT_OFFLINE) — outbound catalog fetch disabled")
    get = opener.open if opener is not None else urllib.request.urlopen
    with get(url, timeout=timeout) as resp:  # noqa: S310 - trusted catalog hosts
        return resp.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# CelesTrak (no auth)
# ---------------------------------------------------------------------------


@dataclass
class CelesTrakClient:
    """CelesTrak GP query client (no authentication required)."""

    base_url: str = "https://celestrak.org/NORAD/elements/gp.php"
    timeout_s: float = 15.0

    def _query(self, **params: str) -> str:
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        return _http_get(url, self.timeout_s)

    def gp_text(self, *, catnr: int | None = None, intldes: str | None = None) -> str:
        """Raw GP response (TLE format) by catalog number or intl designator."""
        if catnr is not None:
            return self._query(CATNR=str(catnr), FORMAT="tle")
        if intldes is not None:
            return self._query(INTDES=intldes, FORMAT="tle")
        raise ValueError("provide either catnr or intldes")

    def tle(self, *, catnr: int | None = None, intldes: str | None = None) -> TlePair | None:
        """Fetch the element set, or None if the object is not catalogued yet."""
        return extract_tle_pair(self.gp_text(catnr=catnr, intldes=intldes))


# ---------------------------------------------------------------------------
# Space-Track (cookie auth)
# ---------------------------------------------------------------------------


@dataclass
class SpaceTrackClient:
    """Space-Track GP query client. Requires an account (identity/password).

    Login establishes a session cookie; queries hit the basicspacedata GP class.
    Credentials are never logged. Intended for programmatic / operator use, not
    the anonymous web path.
    """

    identity: str
    password: str
    base_url: str = "https://www.space-track.org"
    timeout_s: float = 30.0
    _opener: urllib.request.OpenerDirector | None = None

    def login(self) -> None:
        jar = CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        data = urllib.parse.urlencode(
            {"identity": self.identity, "password": self.password}
        ).encode()
        req = urllib.request.Request(f"{self.base_url}/ajaxauth/login", data=data)
        with opener.open(req, timeout=self.timeout_s):  # noqa: S310 - trusted host
            pass
        self._opener = opener

    def gp_json(self, *, catnr: int | None = None, intldes: str | None = None) -> list[dict]:
        """GP records (OMM-JSON) for a catalog number or intl designator."""
        if self._opener is None:
            self.login()
        if catnr is not None:
            pred = f"NORAD_CAT_ID/{catnr}"
        elif intldes is not None:
            pred = f"OBJECT_ID/{intldes}"
        else:
            raise ValueError("provide either catnr or intldes")
        url = f"{self.base_url}/basicspacedata/query/class/gp/{pred}/format/json"
        body = _http_get(url, self.timeout_s, opener=self._opener)
        parsed = json.loads(body) if body.strip() else []
        return parsed if isinstance(parsed, list) else [parsed]

    def tle(self, *, catnr: int | None = None, intldes: str | None = None) -> TlePair | None:
        """Fetch the newest element set, or None if not catalogued yet."""
        records = self.gp_json(catnr=catnr, intldes=intldes)
        if not records:
            return None
        rec = records[0]
        if rec.get("TLE_LINE1") and rec.get("TLE_LINE2"):
            return str(rec["TLE_LINE1"]).rstrip(), str(rec["TLE_LINE2"]).rstrip()
        # Fall back: synthesise the lines from the OMM mean elements.
        from acquisition_platform.ingest.json_ingest import omm_to_tle

        return omm_to_tle(rec)


# ---------------------------------------------------------------------------
# Poller
# ---------------------------------------------------------------------------


@dataclass
class PollResult:
    found: bool
    tle: TlePair | None
    attempts: int
    elapsed_s: float


@dataclass
class CatalogPoller:
    """Retry a catalog fetch until the object appears (or attempts run out).

    `fetch` is any zero-arg callable returning a `TlePair` when the object is
    catalogued and `None` while it is not (e.g.
    `lambda: CelesTrakClient().tle(intldes="2024-001")`). Injecting `fetch`
    (and `sleep`) keeps this loop fully testable offline.
    """

    fetch: Callable[[], TlePair | None]
    interval_s: float = 60.0
    max_attempts: int = 60  # ~1 hour at the 60 s default

    def poll_once(self) -> TlePair | None:
        """One fetch attempt; swallows transient network errors -> None."""
        try:
            return self.fetch()
        except OSError:
            return None

    def wait(
        self,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> PollResult:
        """Block (via `sleep`) until the TLE is found or `max_attempts` elapse."""
        t0 = clock()
        for attempt in range(1, self.max_attempts + 1):
            tle = self.poll_once()
            if tle is not None:
                return PollResult(True, tle, attempt, clock() - t0)
            if attempt < self.max_attempts:
                sleep(self.interval_s)
        return PollResult(False, None, self.max_attempts, clock() - t0)


if __name__ == "__main__":
    sample_tle = (
        "1 25544U 98067A   24001.00000000  .00012345  00000-0  22588-3 0  9990\n"
        "2 25544  51.6400 100.0000 0001000  90.0000 270.0000 15.50000000000010"
    )
    # Response parsing: full response -> pair; junk / empty -> None.
    assert extract_tle_pair(sample_tle) is not None
    assert extract_tle_pair("No GP data found") is None
    assert extract_tle_pair("") is None

    # Poller: object appears on the 3rd attempt; no real sleeping / network.
    state = {"n": 0}

    def fake_fetch() -> TlePair | None:
        state["n"] += 1
        if state["n"] >= 3:
            return extract_tle_pair(sample_tle)
        return None

    ticks = {"t": 0.0}
    poller = CatalogPoller(fetch=fake_fetch, interval_s=5.0, max_attempts=10)
    res = poller.wait(
        sleep=lambda s: ticks.__setitem__("t", ticks["t"] + s), clock=lambda: ticks["t"]
    )
    assert res.found and res.attempts == 3, res
    assert res.tle[0].startswith("1 25544")

    # A never-appearing object exhausts attempts without raising.
    miss = CatalogPoller(fetch=lambda: None, interval_s=1.0, max_attempts=4)
    res2 = miss.wait(sleep=lambda s: None, clock=lambda: 0.0)
    assert not res2.found and res2.attempts == 4

    # Transient network error inside a poll is swallowed to "not found".
    def boom() -> TlePair | None:
        raise OSError("connection refused")

    assert CatalogPoller(fetch=boom).poll_once() is None

    print(f"[catalog] poller found on attempt {res.attempts}, miss attempts={res2.attempts}")
    print("[catalog] PASS")
