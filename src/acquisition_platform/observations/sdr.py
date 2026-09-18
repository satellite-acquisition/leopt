"""Estimate beacon SNR from SDR band power.

``SoapySdrSource`` computes a periodogram from IQ samples and compares the
in-band peak with the median noise floor. This SNR estimate is distinct from
a demodulator's Eb/N0. ``SimulatedSdrSource`` derives reports from pass geometry.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

import numpy as np

from acquisition_platform.observations.base import Observation, ObservationSource

GeometryFn = Callable[[datetime], "tuple[float, float, float]"]


def band_snr_db(
    spectrum_db: np.ndarray,
    freqs_hz: np.ndarray,
    center_hz: float,
    bandwidth_hz: float,
) -> float:
    """Peak-in-band minus the out-of-band median floor (dB), a wideband SNR."""
    in_band = np.abs(freqs_hz - center_hz) <= 0.5 * bandwidth_hz
    if not np.any(in_band):
        return float("-inf")
    peak = float(np.max(spectrum_db[in_band]))
    floor = (
        float(np.median(spectrum_db[~in_band]))
        if np.any(~in_band)
        else float(np.median(spectrum_db))
    )
    return peak - floor


class SoapySdrSource(ObservationSource):
    """Band-power beacon detector over an SDR (needs `SoapySDR` + numpy)."""

    name = "sdr-soapy"
    capabilities = frozenset({"detect", "snr"})

    def __init__(
        self,
        driver: str,
        center_hz: float,
        *,
        sample_rate_hz: float = 2.048e6,
        beacon_offset_hz: float = 0.0,
        beacon_bandwidth_hz: float = 20e3,
        n_samples: int = 1 << 16,
        detect_snr_db: float = 6.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.driver = driver
        self.center_hz = center_hz
        self.sample_rate_hz = sample_rate_hz
        self.beacon_offset_hz = beacon_offset_hz
        self.beacon_bandwidth_hz = beacon_bandwidth_hz
        self.n_samples = n_samples
        self.detect_snr_db = detect_snr_db
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sdr = None
        self._stream = None

    def open(self) -> None:  # pragma: no cover - hardware path
        try:
            import SoapySDR
        except ImportError as exc:
            raise RuntimeError(
                "SoapySdrSource needs the optional 'SoapySDR' package + a radio; use "
                "SimulatedSdrSource for demos/tests."
            ) from exc
        self._sdr = SoapySDR.Device({"driver": self.driver})
        self._sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, self.sample_rate_hz)
        self._sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, self.center_hz)
        self._stream = self._sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        self._sdr.activateStream(self._stream)

    def _read_iq(self) -> np.ndarray:  # pragma: no cover - hardware path
        buff = np.empty(self.n_samples, np.complex64)
        self._sdr.readStream(self._stream, [buff], self.n_samples)
        return buff

    def _spectrum(self, iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        win = np.hanning(len(iq))
        spec = np.fft.fftshift(np.fft.fft(iq * win))
        power_db = 20.0 * np.log10(np.abs(spec) + 1e-12)
        freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), d=1.0 / self.sample_rate_hz))
        return power_db, freqs

    def poll(self) -> Observation | None:  # pragma: no cover - hardware path
        if self._sdr is None:
            self.open()
        power_db, freqs = self._spectrum(self._read_iq())
        snr = band_snr_db(power_db, freqs, self.beacon_offset_hz, self.beacon_bandwidth_hz)
        detected = snr >= self.detect_snr_db
        return Observation(
            t=self._clock(),
            detected=detected,
            locked=detected,
            snr_db=snr if np.isfinite(snr) else None,
            source=self.name,
        )


class SimulatedSdrSource(ObservationSource):
    """Synthesise an SDR band-SNR from pass geometry (no radio required)."""

    name = "sdr-sim"
    capabilities = frozenset({"detect", "snr"})

    def __init__(
        self,
        geometry_fn: GeometryFn,
        antenna_cfg,
        snr_cfg=None,
        *,
        detect_snr_db: float = 6.0,
        noise_floor_db: float = 2.0,
        seed: int = 0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        from antenna_pomdp.models.snr_observation import SnrConfig

        self.geometry_fn = geometry_fn
        self.antenna_cfg = antenna_cfg
        self.snr_cfg = snr_cfg or SnrConfig()
        self.detect_snr_db = detect_snr_db
        self.noise_floor_db = noise_floor_db
        self._rng = np.random.default_rng(seed)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def poll(self) -> Observation | None:
        from antenna_pomdp.models.snr_observation import predicted_ebn0_db

        t = self._clock()
        dtheta, el, rng_m = self.geometry_fn(t)
        # Predicted Eb/N0 is a proxy for the in-band C/N; add estimator noise.
        pred = float(predicted_ebn0_db(dtheta, self.antenna_cfg, el, rng_m, self.snr_cfg))
        snr = pred + float(self._rng.normal(0.0, self.noise_floor_db))
        detected = snr >= self.detect_snr_db
        return Observation(
            t=t,
            detected=detected,
            locked=detected,
            snr_db=snr if detected else None,
            source=self.name,
            meta={"delta_theta_deg": float(np.rad2deg(dtheta))},
        )


if __name__ == "__main__":
    # band_snr_db: a tone 30 dB over a flat floor is recovered near 30 dB.
    freqs = np.linspace(-1e6, 1e6, 4096)
    spec = np.full_like(freqs, 0.0)
    spec[np.argmin(np.abs(freqs - 1e4))] = 30.0
    snr = band_snr_db(spec, freqs, center_hz=1e4, bandwidth_hz=2e4)
    assert 25.0 < snr <= 30.0, snr

    from antenna_pomdp.config import default_config

    cfg = default_config()
    clock = lambda: datetime(2024, 1, 1, tzinfo=timezone.utc)  # noqa: E731
    on = SimulatedSdrSource(
        lambda t: (0.0, np.deg2rad(80.0), 6e5), cfg.antenna, seed=1, clock=clock
    )
    det_rate = np.mean([on.poll().detected for _ in range(30)])
    assert det_rate > 0.9, det_rate
    off = SimulatedSdrSource(
        lambda t: (np.deg2rad(20.0), np.deg2rad(6.0), 2e6), cfg.antenna, seed=2, clock=clock
    )
    det_rate_off = np.mean([off.poll().detected for _ in range(30)])
    assert det_rate_off < 0.2, det_rate_off

    good = next(o for o in (on.poll() for _ in range(10)) if o.detected)
    assert good.snr_db is not None and good.to_snr().ebn0_db is not None  # snr_db stands in

    print(f"[obs.sdr] band SNR={snr:.1f} dB; sim detect on={det_rate:.2f} off={det_rate_off:.2f}")
    print("[obs.sdr] PASS")
