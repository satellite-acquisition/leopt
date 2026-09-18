// Translate console scenarios and observations to the acquisition API.
// The globe uses a local Keplerian model; the backend supplies planned dwells
// and a two-dimensional belief grid, but no orbit-track samples.

(function () {
  'use strict';

  function looksLikeTle(text) {
    if (!text) return false;
    const lines = String(text).trim().split(/\r?\n/).map(s => s.trim());
    const body = lines.filter(l => /^[12]\s/.test(l));
    return body.length >= 2;
  }

  // Console station {name,lat,lng} -> backend StationOut-shaped dict.
  function stationToBackend(s, minElDeg) {
    return {
      name: s.name,
      lat_deg: +s.lat || 0,
      lon_deg: +s.lng || 0,
      alt_m: +s.alt || 0,
      min_el_deg: minElDeg,
    };
  }

  /**
   * Build a PlanRequest body from the console's `config` (see defaultConfig in
   * console.js) and the live `state`.
   * @param {object} config console scenario config
   * @param {object} [state] console React state (for strategy override)
   * @param {object} [opts]  {sat_antenna, n_particles, rollouts, antenna}
   */
  function configToPlanRequest(config, state, opts) {
    config = config || {};
    opts = opts || {};
    const text = config.opm || '';
    const fmt = looksLikeTle(text) ? 'tle' : 'opm';
    const minEl = Math.max(0, +config.minEl || 8);
    const stations = (config.stations || []).map(s => stationToBackend(s, minEl));
    const policy = (state && state.strategy) || config.strategy || 'auto';
    // Known controllers pass straight through; "auto" lets the backend pick
    // info-greedy or PFT-DPW from the antenna-agility heuristic.
    const KNOWN = ['auto', 'pomcp', 'sweep', 'nominal', 'ekf', 'infogreedy', 'pft'];
    return {
      ingest: { format: fmt, text },
      // A custom station list (the console always edits stations directly). To
      // use a built-in network instead, pass its id as opts.network.
      network: opts.network || stations,
      // Ground-antenna spec: beam + mount agility drives the "auto" controller
      // selection on the backend. `mount` maps to a slew-rate preset.
      antenna: opts.antenna || {
        fwhm_deg: Math.max(0.05, +config.beamBeta || 10),
        dwell_s: Math.max(1, +config.holdNom || 10),
        mount: config.mount || 'standard_dish',
      },
      horizon_h: Math.max(0.5, +config.windowH || 6),
      n_particles: opts.n_particles || 200,
      rollouts: opts.rollouts || 150,
      policy: KNOWN.includes(policy) ? policy : 'auto',
      sat_antenna: opts.sat_antenna || config.satAnt || 'turnstile',
    };
  }

  function isoToMs(s) {
    const t = Date.parse(s);
    return Number.isFinite(t) ? t : null;
  }

  /**
   * Convert a PlanResult into console-friendly units. Times are normalised to
   * seconds-since-first-rise so they line up with the console's mission clock.
   * @param {object} plan PlanResult {session_id, passes[], belief_summary{...}}
   */
  function adaptPlan(plan) {
    if (!plan) return null;
    const passes = plan.passes || [];
    // epoch0 = earliest pass rise, so the live timeline starts at t=0 like the sim.
    let epoch0 = Infinity;
    for (const p of passes) {
      const r = isoToMs(p.rise);
      if (r != null && r < epoch0) epoch0 = r;
    }
    if (!Number.isFinite(epoch0)) epoch0 = null;
    const relS = ms => (epoch0 != null && ms != null ? (ms - epoch0) / 1000 : null);

    const adaptedPasses = passes.map(p => {
      const riseS = relS(isoToMs(p.rise));
      const setS = relS(isoToMs(p.set));
      const dwells = (p.pointings || []).map((pt, i) => ({
        idx: i,
        // t_s is engine-relative; expose both the raw value and an absolute
        // estimate (rise + t_s) for the timeline. It is a display alignment,
        // not a shared geometry contract with the local globe simulation.
        tS: pt.t_s,
        tAbsS: riseS != null ? riseS + (pt.t_s || 0) : null,
        az: pt.az_deg,
        el: pt.el_deg,
        pDetect: pt.p_detect,
        holdS: pt.dwell_s,
        dTauS: pt.d_tau_s,
        dThetaDeg: pt.d_theta_deg,
      }));
      return {
        idx: p.idx,
        station: p.station,
        riseISO: p.rise,
        setISO: p.set,
        riseS, setS,
        peakEl: p.peak_el_deg,
        dwells,
      };
    });

    const bs = plan.belief_summary || plan; // /api/observe and /api/plan nest it; /api/belief is flat
    return {
      sessionId: plan.session_id || null,
      entropy: bs.entropy,
      beliefGrid: bs.grid || null, // {along_s, cross_deg, weight}
      passes: adaptedPasses,
      epoch0Ms: epoch0,
      // Which controller is driving, plus the recommendation + rationale (null
      // for the classic baselines). Lets the console explain the auto-selection.
      activePolicy: plan.active_policy || null,
      controller: plan.controller || null,
      raw: plan,
    };
  }

  const LeoptLive = {
    looksLikeTle,
    configToPlanRequest,
    adaptPlan,

    /** Round-trip: build a plan request from config and return the adapted plan. */
    async planFromConfig(config, state, opts) {
      if (!window.LeoptAPI) throw new Error('LeoptLive: LeoptAPI (api.js) not loaded');
      const body = configToPlanRequest(config, state, opts);
      const plan = await window.LeoptAPI.plan(body);
      return adaptPlan(plan);
    },

    /**
     * Preview which controller the backend would auto-select for a config's
     * ground-antenna, without committing to a plan. Returns a ControllerChoice
     * {policy, controller_name, is_agile, rationale, ...}.
     */
    async recommendController(config) {
      if (!window.LeoptAPI) throw new Error('LeoptLive: LeoptAPI (api.js) not loaded');
      return window.LeoptAPI.recommendController({
        antenna: {
          fwhm_deg: Math.max(0.05, +config.beamBeta || 10),
          dwell_s: Math.max(1, +config.holdNom || 10),
          mount: config.mount || 'standard_dish',
        },
      });
    },

    /**
     * Fold one dwell outcome and return the adapted, replanned remainder.
     * @param {object} [report] optional graded RF report from a live source:
     *   {ebn0_db?, snr_db?, locked?, source?}. When it carries an Eb/N0 / SNR the
     *   backend applies the SNR-valued belief update (collapses far faster than a
     *   1-bit detect); omit it for the plain manual DETECT path.
     */
    async observe(sessionId, passIdx, dwellIdx, detected, report) {
      if (!window.LeoptAPI) throw new Error('LeoptLive: LeoptAPI (api.js) not loaded');
      const body = {
        session_id: sessionId, pass_idx: passIdx, dwell_idx: dwellIdx, detected: !!detected,
      };
      if (report) {
        if (report.ebn0_db != null) body.ebn0_db = +report.ebn0_db;
        if (report.snr_db != null) body.snr_db = +report.snr_db;
        if (report.locked != null) body.locked = !!report.locked;
        if (report.source) body.source = String(report.source);
      }
      const plan = await window.LeoptAPI.observe(body);
      return adaptPlan(plan);
    },

    /** GET /api/observation-sources -> the registered live RF-chain feeds. */
    async observationSources() {
      if (!window.LeoptAPI) throw new Error('LeoptLive: LeoptAPI (api.js) not loaded');
      return window.LeoptAPI.req ? window.LeoptAPI.req('GET', '/api/observation-sources')
        : fetch((window.LEOPT_API_BASE || '') + '/api/observation-sources').then(r => r.json());
    },
  };

  if (typeof window !== 'undefined') window.LeoptLive = LeoptLive;
  if (typeof module !== 'undefined' && module.exports) module.exports = LeoptLive;
})();
