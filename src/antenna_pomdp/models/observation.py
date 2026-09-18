"""Receiver-trigger and confirmed-acquisition models for a Gaussian main beam.

The reference-exposure signal-detection probability factorises into a boresight term
and a link term,

    p_signal(Δθ, el, ρ) = P_D(el, ρ) exp(-0.5 (Δθ / σ)²),

where Δθ is the angle between antenna boresight and the true satellite
direction and σ = FWHM / 2.355. A receiver can also produce an off-target raw
trigger with probability f:

    p_trigger = p_signal + (1 - p_signal) f.

A raw trigger is not synonymous with spacecraft acquisition. A carrier and
identity confirmation stage accepts signal-caused and false triggers with
separately configurable probabilities. Either acceptance terminates the
binary-observation model, but evaluation keeps *correct acquisition* and
*false confirmation* as distinct competing outcomes. This prevents a
per-dwell false-alarm floor—or an imperfect confirmation stage—from being
counted as mission success.

`P_D(el, ρ)` is the on-axis probability over
`observation_reference_time_s`. With the default config
(`link_budget_enabled = False`) it collapses to the constant `1 - miss_rate`,
giving a constant on-axis detection probability. When the link budget is
enabled, `P_D` is computed from a compact, self-contained S-band link model
(free-space path loss vs. slant range ρ, plus a slant atmospheric attenuation
that grows as 1/sin(el)). Confirmation outcomes are independent Bernoulli
draws in the present model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from antenna_pomdp.config import AntennaConfig, beam_sigma_rad
from antenna_pomdp.orbit.geometry import angular_separation

# Floor on elevation used in the 1/sin(el) airmass term, so the slant
# atmospheric loss stays finite for grazing (near-zero-elevation) geometry.
_MIN_LINK_ELEVATION_RAD = np.deg2rad(1.0)


def on_axis_pd(
    antenna_cfg: AntennaConfig,
    elevation_rad: float | np.ndarray | None = None,
    range_m: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """On-axis (Δθ = 0) probability of detection P_D(el, ρ).

    If the link budget is disabled, or no geometry is supplied, returns the
    constant `1 - miss_rate` (original model). Otherwise maps an S-band link
    margin to a detection probability through a logistic curve:

        margin = ref_margin_db
                 - 20 log10(ρ / ref_range_m)           # free-space path loss
                 - zenith_atmos_loss_db (1/sin el - 1)  # excess slant airmass
        P_D    = pd_max / (1 + exp(-pd_slope (margin - pd_margin50_db)))
    """
    if not antenna_cfg.link_budget_enabled or elevation_rad is None or range_m is None:
        return 1.0 - antenna_cfg.miss_rate

    el = np.maximum(elevation_rad, _MIN_LINK_ELEVATION_RAD)
    fspl_excess_db = 20.0 * np.log10(np.asarray(range_m, dtype=float) / antenna_cfg.ref_range_m)
    atmos_excess_db = antenna_cfg.zenith_atmos_loss_db * (1.0 / np.sin(el) - 1.0)
    margin_db = antenna_cfg.ref_margin_db - fspl_excess_db - atmos_excess_db
    pd = antenna_cfg.pd_max / (
        1.0 + np.exp(-antenna_cfg.pd_slope_per_db * (margin_db - antenna_cfg.pd_margin50_db))
    )
    return pd


@dataclass(frozen=True)
class Pointing:
    """A commanded boresight direction in topocentric (az, el) coordinates, radians."""

    az: float
    el: float


def detection_probability(
    delta_theta_rad: float | np.ndarray,
    antenna_cfg: AntennaConfig,
    elevation_rad: float | np.ndarray | None = None,
    range_m: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """P(terminal confirmation | geometry).

    The historical public name is retained for API compatibility. New code
    should prefer :func:`terminal_confirmation_probability`.
    """
    return terminal_confirmation_probability(delta_theta_rad, antenna_cfg, elevation_rad, range_m)


def signal_detection_probability(
    delta_theta_rad: float | np.ndarray,
    antenna_cfg: AntennaConfig,
    elevation_rad: float | np.ndarray | None = None,
    range_m: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """P(signal-caused receiver detection over the configured dwell)."""
    sigma = beam_sigma_rad(antenna_cfg)
    pd_on = on_axis_pd(antenna_cfg, elevation_rad, range_m)
    p_signal_ref = np.clip(
        pd_on * np.exp(-0.5 * (delta_theta_rad / sigma) ** 2),
        0.0,
        1.0,
    )
    if elevation_rad is not None:
        visible = np.asarray(elevation_rad) >= np.deg2rad(antenna_cfg.min_signal_elevation_deg)
        p_signal_ref = np.where(visible, p_signal_ref, 0.0)
    return _scale_single_reference_probability(p_signal_ref, antenna_cfg)


def reference_event_probabilities(
    delta_theta_rad: float | np.ndarray,
    antenna_cfg: AntennaConfig,
    elevation_rad: float | np.ndarray | None = None,
    range_m: float | np.ndarray | None = None,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Correct/false terminal probabilities over the reference exposure."""
    sigma = beam_sigma_rad(antenna_cfg)
    pd_on = on_axis_pd(antenna_cfg, elevation_rad, range_m)
    p_signal_ref = np.clip(
        pd_on * np.exp(-0.5 * (delta_theta_rad / sigma) ** 2),
        0.0,
        1.0,
    )
    if elevation_rad is not None:
        visible = np.asarray(elevation_rad) >= np.deg2rad(antenna_cfg.min_signal_elevation_deg)
        p_signal_ref = np.where(visible, p_signal_ref, 0.0)
    p_true_ref = p_signal_ref * float(
        np.clip(antenna_cfg.confirmation_true_accept_rate, 0.0, 1.0)
    )
    p_false_ref = (
        (1.0 - p_signal_ref)
        * float(np.clip(antenna_cfg.false_alarm_rate, 0.0, 1.0))
        * float(np.clip(antenna_cfg.confirmation_false_accept_rate, 0.0, 1.0))
    )
    return (
        np.clip(p_true_ref, 0.0, 1.0),
        np.clip(p_false_ref, 0.0, 1.0),
    )


def _exposure_ratio(antenna_cfg: AntennaConfig) -> float:
    reference = float(antenna_cfg.observation_reference_time_s)
    dwell = float(antenna_cfg.dwell_time_s)
    if reference <= 0.0 or dwell <= 0.0:
        raise ValueError("observation reference time and dwell time must be positive")
    return dwell / reference


def _scale_single_reference_probability(
    probability: float | np.ndarray,
    antenna_cfg: AntennaConfig,
) -> float | np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    ratio = _exposure_ratio(antenna_cfg)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = -np.expm1(ratio * np.log1p(-p))
    scaled = np.where(p >= 1.0, 1.0, scaled)
    if np.ndim(probability) == 0:
        return float(scaled)
    return np.clip(scaled, 0.0, 1.0)


def scale_competing_reference_probabilities(
    p_true_ref: float | np.ndarray,
    p_false_ref: float | np.ndarray,
    antenna_cfg: AntennaConfig,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Scale competing reference-exposure risks to the configured dwell.

    The reference probabilities are converted to a total continuous hazard,
    split between causes in their reference proportions, and integrated over
    `dwell_time_s`. At the reference dwell this exactly returns the inputs.
    """
    p_true = np.asarray(p_true_ref, dtype=float)
    p_false = np.asarray(p_false_ref, dtype=float)
    total_ref = np.clip(p_true + p_false, 0.0, 1.0)
    total = np.asarray(_scale_single_reference_probability(total_ref, antenna_cfg))
    fraction_true = np.divide(
        p_true,
        total_ref,
        out=np.zeros_like(total_ref, dtype=float),
        where=total_ref > 0.0,
    )
    scaled_true = np.clip(total * fraction_true, 0.0, 1.0)
    scaled_false = np.clip(total - scaled_true, 0.0, 1.0)
    if np.ndim(p_true_ref) == 0 and np.ndim(p_false_ref) == 0:
        return float(scaled_true), float(scaled_false)
    return scaled_true, scaled_false


def receiver_trigger_probability(
    delta_theta_rad: float | np.ndarray,
    antenna_cfg: AntennaConfig,
    elevation_rad: float | np.ndarray | None = None,
    range_m: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """P(raw receiver trigger | geometry), including off-target false alarms."""
    sigma = beam_sigma_rad(antenna_cfg)
    pd_on = on_axis_pd(antenna_cfg, elevation_rad, range_m)
    p_signal_ref = np.clip(
        pd_on * np.exp(-0.5 * (delta_theta_rad / sigma) ** 2),
        0.0,
        1.0,
    )
    if elevation_rad is not None:
        visible = np.asarray(elevation_rad) >= np.deg2rad(antenna_cfg.min_signal_elevation_deg)
        p_signal_ref = np.where(visible, p_signal_ref, 0.0)
    p_fa = float(np.clip(antenna_cfg.false_alarm_rate, 0.0, 1.0))
    p_trigger_ref = p_signal_ref + (1.0 - p_signal_ref) * p_fa
    return _scale_single_reference_probability(p_trigger_ref, antenna_cfg)


def true_acquisition_probability(
    delta_theta_rad: float | np.ndarray,
    antenna_cfg: AntennaConfig,
    elevation_rad: float | np.ndarray | None = None,
    range_m: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """P(signal-caused trigger is correctly confirmed | geometry)."""
    p_true_ref, p_false_ref = reference_event_probabilities(
        delta_theta_rad,
        antenna_cfg,
        elevation_rad,
        range_m,
    )
    p_true, _ = scale_competing_reference_probabilities(
        p_true_ref,
        p_false_ref,
        antenna_cfg,
    )
    return p_true


def false_confirmation_probability(
    delta_theta_rad: float | np.ndarray,
    antenna_cfg: AntennaConfig,
    elevation_rad: float | np.ndarray | None = None,
    range_m: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """P(off-target false trigger is incorrectly accepted | geometry)."""
    p_true_ref, p_false_ref = reference_event_probabilities(
        delta_theta_rad,
        antenna_cfg,
        elevation_rad,
        range_m,
    )
    _, p_false = scale_competing_reference_probabilities(
        p_true_ref,
        p_false_ref,
        antenna_cfg,
    )
    return p_false


def terminal_confirmation_probability(
    delta_theta_rad: float | np.ndarray,
    antenna_cfg: AntennaConfig,
    elevation_rad: float | np.ndarray | None = None,
    range_m: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """P(correct acquisition or false confirmation | geometry)."""
    return np.clip(
        true_acquisition_probability(delta_theta_rad, antenna_cfg, elevation_rad, range_m)
        + false_confirmation_probability(delta_theta_rad, antenna_cfg, elevation_rad, range_m),
        0.0,
        1.0,
    )


def confirmed_acquisition_probability(
    delta_theta_rad: float | np.ndarray,
    antenna_cfg: AntennaConfig,
    elevation_rad: float | np.ndarray | None = None,
    range_m: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """Compatibility alias for the binary model's terminal confirmation event.

    This historical name is unambiguous only when false acceptance is zero.
    Use :func:`true_acquisition_probability` for mission success and
    :func:`false_confirmation_probability` for its competing failure event.
    """
    return terminal_confirmation_probability(delta_theta_rad, antenna_cfg, elevation_rad, range_m)


def observation_likelihood(
    detected: bool,
    delta_theta_rad: float | np.ndarray,
    antenna_cfg: AntennaConfig,
    elevation_rad: float | np.ndarray | None = None,
    range_m: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """P(o | s, a) for o ∈ {terminal confirmation, no confirmation}."""
    p = detection_probability(delta_theta_rad, antenna_cfg, elevation_rad, range_m)
    return p if detected else 1.0 - p


def sample_observation(
    delta_theta_rad: float,
    antenna_cfg: AntennaConfig,
    rng: np.random.Generator,
    elevation_rad: float | None = None,
    range_m: float | None = None,
) -> bool:
    """Bernoulli draw of a terminal-confirmation / no-confirmation outcome."""
    p = detection_probability(delta_theta_rad, antenna_cfg, elevation_rad, range_m)
    return bool(rng.random() < p)


def delta_theta_from_azel(pointing: Pointing, target_az: float, target_el: float) -> float:
    """Angular separation between commanded boresight and a target az/el."""
    return angular_separation(pointing.az, pointing.el, target_az, target_el)


if __name__ == "__main__":
    from dataclasses import replace

    from antenna_pomdp.config import default_config

    cfg = default_config()
    rng = np.random.default_rng(0)

    # On-boresight confirmed acquisition follows the signal term when false
    # triggers are perfectly rejected.
    p_on = float(detection_probability(0.0, cfg.antenna))
    expected = (1.0 - cfg.antenna.miss_rate) * cfg.antenna.confirmation_true_accept_rate
    assert abs(p_on - expected) < 1e-6, (p_on, expected)

    # Far-off boresight: a raw false trigger remains possible but cannot be
    # counted as acquisition under the default confirmation model.
    p_far = float(detection_probability(np.deg2rad(60.0), cfg.antenna))
    p_trigger_far = float(receiver_trigger_probability(np.deg2rad(60.0), cfg.antenna))
    assert p_far < 1e-6, p_far
    assert abs(p_trigger_far - cfg.antenna.false_alarm_rate) < 1e-3, p_trigger_far

    # Monte Carlo at boresight ≈ p_on.
    samples = [sample_observation(0.0, cfg.antenna, rng) for _ in range(5000)]
    hit_rate = np.mean(samples)
    assert abs(hit_rate - p_on) < 0.03, (hit_rate, p_on)

    # Link budget: a high/near pass must be easier than a low/far horizon pass.
    lb_cfg = replace(cfg.antenna, link_budget_enabled=True)
    pd_zenith = float(on_axis_pd(lb_cfg, np.deg2rad(85.0), 600e3))
    pd_horizon = float(on_axis_pd(lb_cfg, np.deg2rad(5.0), 1900e3))
    assert pd_zenith > pd_horizon, (pd_zenith, pd_horizon)
    assert pd_zenith <= lb_cfg.pd_max + 1e-9
    # With the link budget off, on-axis P_D is the constant model.
    pd_const = float(on_axis_pd(replace(cfg.antenna, link_budget_enabled=False), 0.1, 1e6))
    assert abs(pd_const - (1.0 - cfg.antenna.miss_rate)) < 1e-9, pd_const

    print(
        f"[observation] p_confirm(0)={p_on:.3f}, p_confirm(60deg)={p_far:.3f}, "
        f"p_trigger(60deg)={p_trigger_far:.3f}, "
        f"MC hit rate={hit_rate:.3f}"
    )
    print(f"[observation] link P_D: zenith={pd_zenith:.3f}, horizon={pd_horizon:.3f}")
    print("[observation] PASS")
