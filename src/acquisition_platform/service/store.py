"""SQLite session checkpoints and a hash-chained audit log.

Particles are stored as TLE lines and launch offsets, then rebuilt on load.
The snapshot also records RNG state. Passes, plans, and solver objects are
recomputed rather than serialized.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np

from acquisition_platform.planner.controller_select import GroundAntennaSpec
from acquisition_platform.planner.session import PlanSession
from acquisition_platform.safety import SafetyConfig, WatchdogConfig
from acquisition_platform.safety.envelope import KeepOutBox
from acquisition_platform.satellite_antennas import get_sat_antenna
from antenna_pomdp.models.doppler import FrequencyConfig
from antenna_pomdp.config import (
    AntennaConfig,
    Config,
    EvalConfig,
    FilterConfig,
    GroundStationNetworkConfig,
    HeuristicConfig,
    KoopmanConfig,
    LeopConfig,
    OrbitConfig,
    PomcpConfig,
    SolverConfig,
    StationConfig,
)
from antenna_pomdp.models.particle_filter import ParticleBelief
from antenna_pomdp.orbit.propagator import SGP4Propagator
from antenna_pomdp.orbit.sampler import OrbitParticle

SCHEMA_VERSION = 1
_GENESIS_HASH = "0" * 64


# ---------------------------------------------------------------------------
# Belief (de)serialization — (l1, l2, slip) triples, no Satrec pickling
# ---------------------------------------------------------------------------


def belief_to_dict(belief: ParticleBelief) -> dict:
    return {
        "weights": [float(w) for w in belief.weights],
        "particles": [
            [p.propagator.tle_line1, p.propagator.tle_line2, float(p.launch_slip_s)]
            for p in belief.particles
        ],
    }


def belief_from_dict(d: dict) -> ParticleBelief:
    particles = [
        OrbitParticle(SGP4Propagator(l1, l2), float(slip)) for (l1, l2, slip) in d["particles"]
    ]
    return ParticleBelief(particles=particles, weights=np.asarray(d["weights"], dtype=float))


# ---------------------------------------------------------------------------
# Config reconstruction (from __future__ annotations => explicit, not reflective)
# ---------------------------------------------------------------------------


def _config_from_dict(d: dict) -> Config:
    return Config(
        orbit=OrbitConfig(**d["orbit"]),
        station=StationConfig(**d["station"]),
        antenna=AntennaConfig(**d["antenna"]),
        filter=FilterConfig(**d["filter"]),
        pomcp=PomcpConfig(**d["pomcp"]),
        solver=SolverConfig(**d["solver"]),
        heuristic=HeuristicConfig(**d["heuristic"]),
        koopman=KoopmanConfig(**d["koopman"]),
        eval=EvalConfig(**d["eval"]),
        leop=LeopConfig(**d["leop"]),
    )


def _network_from_dict(d: dict) -> GroundStationNetworkConfig:
    return GroundStationNetworkConfig(
        network_id=d["network_id"],
        name=d["name"],
        stations=tuple(StationConfig(**s) for s in d["stations"]),
        station_ids=tuple(d.get("station_ids", ())),
    )


def _safety_from_dict(d: dict) -> SafetyConfig:
    keep = tuple(KeepOutBox(**b) for b in d.get("keep_out", ()))
    return SafetyConfig(**{**d, "keep_out": keep})


# ---------------------------------------------------------------------------
# Session snapshot
# ---------------------------------------------------------------------------


def snapshot_session(session: PlanSession) -> dict:
    """Serialize everything needed to rebuild a session faithfully (JSON-safe)."""
    wd = session.watchdog
    return {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(session.config),
        "network": asdict(session.network),
        "safety": asdict(session.safety),
        "watchdog_cfg": asdict(session.watchdog_cfg),
        "freq_cfg": asdict(session.freq_cfg),
        "ground_antenna": (
            asdict(session.ground_antenna) if session.ground_antenna is not None else None
        ),
        "policy": session.policy,
        "policy_before_fallback": session._policy_before_fallback,
        "fallback_reason": session.fallback_reason,
        "authority_level": session.authority_level,
        "sat_antenna_id": session.sat_antenna.id,
        # leop_start is MUTATED on observe() — persist it, never recompute from epoch.
        "leop_start": session.leop_start.isoformat(),
        "horizon_h": session._horizon_h,
        "belief": belief_to_dict(session.belief),
        "rng_state": session.rng.bit_generator.state,
        "watchdog_state": {
            "consecutive_divergence": wd.consecutive_divergence,
            "acquired": wd.acquired,
            "latched_open_loop": wd.latched_open_loop,
            "last_reason": wd.last_reason,
        },
        "obs_outcomes": [bool(x) for x in session._obs_outcomes],
    }


def session_from_snapshot(snap: dict) -> PlanSession:
    """Rebuild a `PlanSession` from a snapshot and regenerate its plan."""
    if snap.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported snapshot schema {snap.get('schema_version')!r}")
    config = _config_from_dict(snap["config"])
    network = _network_from_dict(snap["network"])
    belief = belief_from_dict(snap["belief"])
    ground = GroundAntennaSpec(**snap["ground_antenna"]) if snap.get("ground_antenna") else None
    rng = np.random.default_rng()
    rng.bit_generator.state = snap["rng_state"]
    leop_start = datetime.fromisoformat(snap["leop_start"])

    wds = snap["watchdog_state"]
    latched = bool(wds.get("latched_open_loop"))
    # Construct with the pre-fallback controller when latched, so controller_choice
    # is preserved; then re-apply the fallback state below.
    effective_policy = (
        snap.get("policy_before_fallback")
        if latched and snap.get("policy_before_fallback")
        else snap["policy"]
    )

    session = PlanSession(
        config=config,
        network=network,
        belief=belief,
        policy=effective_policy,
        sat_antenna=get_sat_antenna(snap["sat_antenna_id"]),
        ground_antenna=ground,
        authority_level=snap["authority_level"],
        safety=_safety_from_dict(snap["safety"]),
        watchdog_cfg=WatchdogConfig(**snap["watchdog_cfg"]),
        freq_cfg=FrequencyConfig(**snap["freq_cfg"]) if snap.get("freq_cfg") else FrequencyConfig(),
        leop_start=leop_start,
        rng=rng,
    )
    # Restore mutable runtime state.
    session._obs_outcomes = list(snap.get("obs_outcomes", []))
    session.watchdog.consecutive_divergence = int(wds.get("consecutive_divergence", 0))
    session.watchdog.acquired = bool(wds.get("acquired", False))
    if latched:
        session.watchdog.latched_open_loop = True
        session.watchdog.last_reason = wds.get("last_reason", "")
        session._policy_before_fallback = snap.get("policy_before_fallback")
        session.fallback_reason = snap.get("fallback_reason", "")
        session.policy = "sweep"
    # Regenerate the (unpersisted) plan from the restored belief.
    session.plan_network(float(snap["horizon_h"]))
    return session


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class SessionStore:
    """SQLite-backed session persistence + hash-chained audit log."""

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        # check_same_thread=False + a lock: FastAPI serves requests from a thread
        # pool, but this is a single-writer, human-paced store, so one connection
        # guarded by a lock is both correct and simple.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.row_factory = sqlite3.Row
        # session_ids skipped by the last load_all_sessions (un-rebuildable snapshots).
        self.skipped: list[str] = []
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        c = self._conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                tenant_id   TEXT NOT NULL DEFAULT 'default',
                updated_ts  TEXT NOT NULL,
                snapshot    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit (
                seq        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         TEXT NOT NULL,
                tenant_id  TEXT NOT NULL DEFAULT 'default',
                actor      TEXT NOT NULL DEFAULT 'system',
                role       TEXT NOT NULL DEFAULT 'operator',
                session_id TEXT,
                action     TEXT NOT NULL,
                detail     TEXT NOT NULL DEFAULT '{}',
                prev_hash  TEXT NOT NULL,
                curr_hash  TEXT NOT NULL UNIQUE
            );
            CREATE TRIGGER IF NOT EXISTS audit_no_update
                BEFORE UPDATE ON audit
                BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
            CREATE TRIGGER IF NOT EXISTS audit_no_delete
                BEFORE DELETE ON audit
                BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
            """
        )
        c.commit()

    # -- sessions ----------------------------------------------------------
    def save_session(
        self, session_id: str, session: PlanSession, tenant_id: str = "default"
    ) -> None:
        snap = _canonical(snapshot_session(session))
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions(session_id, tenant_id, updated_ts, snapshot) VALUES(?,?,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET updated_ts=excluded.updated_ts, "
                "snapshot=excluded.snapshot, tenant_id=excluded.tenant_id",
                (session_id, tenant_id, ts, snap),
            )
            self._conn.commit()

    def load_session(self, session_id: str) -> PlanSession | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT snapshot FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        return session_from_snapshot(json.loads(row["snapshot"]))

    def load_all_sessions(self) -> dict[str, PlanSession]:
        with self._lock:
            rows = self._conn.execute("SELECT session_id, snapshot FROM sessions").fetchall()
        out: dict[str, PlanSession] = {}
        self.skipped = []
        for row in rows:
            try:
                out[row["session_id"]] = session_from_snapshot(json.loads(row["snapshot"]))
            except Exception:  # noqa: BLE001 - skip ANY un-rebuildable snapshot
                # session_from_snapshot re-runs plan_network, which can raise a
                # RuntimeError from SGP4; one bad checkpoint must not abort the
                # recovery of every other session.
                self.skipped.append(row["session_id"])
        return out

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
            self._conn.commit()

    def session_ids(self) -> list[str]:
        with self._lock:
            return [r["session_id"] for r in self._conn.execute("SELECT session_id FROM sessions")]

    # -- audit -------------------------------------------------------------
    def append_audit(
        self,
        action: str,
        session_id: str | None = None,
        actor: str = "system",
        role: str = "operator",
        tenant_id: str = "default",
        detail: dict | None = None,
    ) -> str:
        """Append one hash-chained audit record; return its curr_hash."""
        with self._lock:
            prev = self._conn.execute(
                "SELECT curr_hash FROM audit ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            prev_hash = prev["curr_hash"] if prev else _GENESIS_HASH
            ts = datetime.now(timezone.utc).isoformat()
            detail_json = _canonical(detail or {})
            payload = {
                "ts": ts,
                "tenant_id": tenant_id,
                "actor": actor,
                "role": role,
                "session_id": session_id,
                "action": action,
                "detail": detail_json,
            }
            curr_hash = hashlib.sha256((_canonical(payload) + prev_hash).encode()).hexdigest()
            self._conn.execute(
                "INSERT INTO audit(ts, tenant_id, actor, role, session_id, action, detail, "
                "prev_hash, curr_hash) VALUES(?,?,?,?,?,?,?,?,?)",
                (ts, tenant_id, actor, role, session_id, action, detail_json, prev_hash, curr_hash),
            )
            self._conn.commit()
        return curr_hash

    def verify_chain(self) -> int | None:
        """Re-walk the audit chain; return the first tampered `seq`, or None if intact."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, ts, tenant_id, actor, role, session_id, action, detail, prev_hash, "
                "curr_hash FROM audit ORDER BY seq ASC"
            ).fetchall()
        prev_hash = _GENESIS_HASH
        for row in rows:
            payload = {
                "ts": row["ts"],
                "tenant_id": row["tenant_id"],
                "actor": row["actor"],
                "role": row["role"],
                "session_id": row["session_id"],
                "action": row["action"],
                "detail": row["detail"],
            }
            expect = hashlib.sha256((_canonical(payload) + prev_hash).encode()).hexdigest()
            if row["prev_hash"] != prev_hash or row["curr_hash"] != expect:
                return int(row["seq"])
            prev_hash = row["curr_hash"]
        return None

    def audit_log(self, session_id: str | None = None, limit: int = 200) -> list[dict]:
        with self._lock:
            if session_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM audit ORDER BY seq DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM audit WHERE session_id=? ORDER BY seq DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()


if __name__ == "__main__":
    from dataclasses import replace

    from acquisition_platform.estimator.ukf import SgpUkf
    from acquisition_platform.ingest.tle import parse_tle
    from antenna_pomdp.config import default_config, network_single_university

    SSO = (
        "1 99999U 24001A   24001.00000000  .00000000  00000-0  00000-0 0  9990\n"
        "2 99999  97.4000 100.0000 0001000  90.0000 270.0000 15.20000000000017"
    )
    base = default_config()
    l1, l2 = SSO.splitlines()
    cfg = replace(
        base,
        orbit=replace(base.orbit, nominal_tle_line1=l1, nominal_tle_line2=l2),
        antenna=replace(base.antenna, fwhm_deg=10.0, dwell_time_s=10.0),
        filter=replace(base.filter, n_particles=48),
    )
    rng = np.random.default_rng(0)
    cov = np.diag([3000.0, 1000.0, 1000.0, 3.0, 3.0, 3.0]) ** 2
    opm, _, _ = parse_tle(SSO, cov_6x6=cov, frame="RTN")
    ukf = SgpUkf(orbit_cfg=cfg.orbit)
    belief = ukf.seed_particles(opm, n=cfg.filter.n_particles, rng=rng)
    sess = PlanSession(
        config=cfg,
        network=network_single_university(),
        belief=belief,
        policy="infogreedy",
        leop_start=opm.epoch,
        ukf=ukf,
        rng=rng,
    )
    plans = sess.plan_network(horizon_h=8.0)
    first = next(p for p in plans if p.pointings)
    sess.observe(first.idx, 0, detected=False)

    store = SessionStore(":memory:")
    store.append_audit("session.create", session_id="sid-1", detail={"policy": "infogreedy"})
    store.save_session("sid-1", sess)

    # Simulate a restart: reload and compare belief fidelity.
    restored = store.load_session("sid-1")
    assert restored is not None
    assert np.allclose(restored.belief.weights, sess.belief.weights)
    assert restored.leop_start == sess.leop_start
    assert [p.launch_slip_s for p in restored.belief.particles] == [
        p.launch_slip_s for p in sess.belief.particles
    ]

    def _plan_fingerprint(s):
        return [(round(pt.az_rad, 9), round(pt.el_rad, 9)) for p in s._plans for pt in p.pointings]

    # Determinism: the snapshotted RNG state makes re-planning reproducible — two
    # independent restores produce byte-identical schedules.
    restored2 = store.load_session("sid-1")
    assert _plan_fingerprint(restored) == _plan_fingerprint(restored2)

    # Audit chain intact.
    store.append_audit("observe", session_id="sid-1", detail={"detected": False})
    assert store.verify_chain() is None

    # Tamper detection: drop the guard, mutate a row, re-check.
    store._conn.execute("DROP TRIGGER audit_no_update")
    store._conn.execute("UPDATE audit SET action='forged' WHERE seq=1")
    store._conn.commit()
    assert store.verify_chain() == 1
    print("[store] restart round-trip OK; tamper detected at seq=1")
    print("[store] PASS")
