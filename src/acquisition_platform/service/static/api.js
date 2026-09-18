// Browser client for the acquisition API. Angles are degrees and times are seconds.
(function () {
  'use strict';

  // Same-origin by default (the console is served by the same FastAPI app).
  // Override with `window.LEOPT_API_BASE = 'http://host:port'` before load.
  function base() {
    return (window.LEOPT_API_BASE || '').replace(/\/$/, '');
  }

  async function req(method, path, body) {
    const opts = { method, headers: {} };
    if (body !== undefined) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(base() + path, opts);
    const text = await res.text();
    let data = null;
    if (text) {
      try { data = JSON.parse(text); } catch (_e) { data = text; }
    }
    if (!res.ok) {
      const detail = data && data.detail ? data.detail : (text || res.statusText);
      const err = new Error('LeoptAPI ' + method + ' ' + path + ' -> ' + res.status + ': ' + detail);
      err.status = res.status;
      err.body = data;
      throw err;
    }
    return data;
  }

  const LeoptAPI = {
    /** GET /api/networks -> [{id,name,n_stations,stations:[{name,lat_deg,lon_deg,alt_m,min_el_deg}]}] */
    networks() { return req('GET', '/api/networks'); },

    /** GET /api/sat-antennas -> [{id,name,description,coverage,base_hold_s}] */
    satAntennas() { return req('GET', '/api/sat-antennas'); },

    /** GET /api/mounts -> [{id,name,slew_rate_deg_s,agile,description}] ground-antenna presets */
    mounts() { return req('GET', '/api/mounts'); },

    /**
     * POST /api/recommend-controller -> ControllerChoice
     * Preview which controller `policy:'auto'` would pick for an antenna, with a
     * plain-language rationale — before committing to a plan.
     * @param {{antenna:object, sat_antenna?:string}} body
     */
    recommendController(body) { return req('POST', '/api/recommend-controller', body); },

    /**
     * POST /api/ingest -> IngestResult
     * @param {{format:'opm'|'tle', text:string, covariance?:number[][], frame?:'RTN'|'ECI'}} body
     */
    ingest(body) { return req('POST', '/api/ingest', body); },

    /**
     * POST /api/plan -> PlanResult {session_id, passes[], belief_summary{entropy,grid}}
     * @param {object} body PlanRequest (see schemas.PlanRequest). `ingest` may be a
     *   raw {format,text,...} body OR a fully-formed IngestResult.
     */
    plan(body) { return req('POST', '/api/plan', body); },

    /**
     * POST /api/observe -> PlanResult (the replanned remainder)
     * @param {{session_id:string, pass_idx:number, dwell_idx:number, detected:boolean}} body
     */
    observe(body) { return req('POST', '/api/observe', body); },

    /** GET /api/belief/{session_id} -> {entropy, grid:{along_s,cross_deg,weight}} */
    belief(sessionId) { return req('GET', '/api/belief/' + encodeURIComponent(sessionId)); },

    /**
     * POST /api/authority -> PlanResult
     * Change the graduated-trust authority level (advisory..autonomous).
     * @param {{session_id:string, level:'L0_ADVISORY'|'L1_SHADOW'|'L2_SUPERVISED'|'L3_AUTONOMOUS'}} body
     */
    setAuthority(body) { return req('POST', '/api/authority', body); },

    /** POST /api/rearm -> PlanResult (restore the controller after a watchdog fallback) */
    rearm(sessionId) { return req('POST', '/api/rearm', { session_id: sessionId }); },

    /** GET /api/audit/{session_id} -> {enabled, entries[], chain_ok} command history */
    audit(sessionId) { return req('GET', '/api/audit/' + encodeURIComponent(sessionId)); },

    /**
     * POST /api/catalog/tle -> {found, line1?, line2?, source, message}
     * One catalog fetch attempt; poll on an interval until found=true, then
     * switch the belief from the OPM/OEM prior to the pulled TLE.
     * @param {{source?:'celestrak'|'spacetrack', catnr?:number, intldes?:string,
     *          identity?:string, password?:string}} body
     */
    catalogTle(body) { return req('POST', '/api/catalog/tle', body); },

    /** GET /api/observation-sources -> [{id,name,transport,needs_hardware,capabilities,description}] */
    observationSources() { return req('GET', '/api/observation-sources'); },

    /** GET /api/drivers -> [{id,name,transport,capabilities,description}] */
    drivers() { return req('GET', '/api/drivers'); },

    /**
     * POST /api/export/track -> {format, filename, n_samples, content}
     * Densify a planned pass into a continuous track/ephemeris file for an ACU.
     * @param {{session_id:string, pass_idx:number, format?:'csv'|'ccsds_pointing'|'oem',
     *          rate_hz?:number, object_name?:string}} body
     */
    exportTrack(body) { return req('POST', '/api/export/track', body); },

    /**
     * POST /api/export/schedule -> one atomic JSON/PDF representation pair.
     * The server captures its materialized backend plan once, renders both
     * files from that same schedule revision, and returns the PDF as base64.
     * @param {{session_id:string}} body
     * @returns {Promise<{schema_version:string, schedule_id:string,
     *   json_filename:string, json_content:string, json_sha256:string,
     *   json_size_bytes:number, pdf_filename:string, pdf_base64:string,
     *   pdf_sha256:string, pdf_size_bytes:number, pass_count:number,
     *   dwell_count:number, complete:boolean}>}
     */
    exportSchedule(body) { return req('POST', '/api/export/schedule', body); },

    // Low-level request helper, exposed so thin callers (live.js) can reuse it.
    req,
  };

  // CommonJS exports support browser-client tests.
  if (typeof window !== 'undefined') window.LeoptAPI = LeoptAPI;
  if (typeof module !== 'undefined' && module.exports) module.exports = LeoptAPI;
})();
