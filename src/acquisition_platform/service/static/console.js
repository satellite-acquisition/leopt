'use strict';

class LeoptConsole extends React.Component {
  state = {
    t: 0, playing: false, mode: 'plan', splashVisible: true, rolledOut: false, launching: false, splashShown: false, revealed: false, dragOver: false,
    truthDeg: this.props.truthDeg ?? 1.8,
    strategy: this.props.strategy ?? 'auto',
    treatment: this.props.beliefTreatment ?? 'heatmap',
    stepIndex: 0, acquired: false, acqT: null, advOpen: false, advTab: 'orbit', tickN: 0,
    scheduleExportBusy: false, scheduleExportStatus: '', scheduleExportError: false
  };

  componentDidMount() { this.setup(); this.initGlobe(); this.initSplashGlobe(); setTimeout(() => this.setState({ splashShown: true }), 90); setTimeout(() => this.previewController(), 500); }
  componentWillUnmount() { clearInterval(this._timer); clearInterval(this._auto); cancelAnimationFrame(this._raf); cancelAnimationFrame(this._sraf); if (this._ro) this._ro.disconnect(); }

  async initSplashGlobe() {
    await this.waitFor(() => window.THREE && this.splashGlobeEl && this.splashGlobeEl.clientWidth > 0, 6000);
    const T = window.THREE; if (!T || !this.splashGlobeEl || this._splashGlobe) return;
    const el = this.splashGlobeEl, S = Math.min(el.clientWidth, el.clientHeight) || 600, R = 100;
    const renderer = new T.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1)); renderer.setSize(S, S); renderer.setClearColor(0x000000, 0);
    renderer.domElement.style.cssText = 'display:block;width:100%;height:100%'; el.appendChild(renderer.domElement);
    const scene = new T.Scene(), camera = new T.PerspectiveCamera(38, 1, 1, 4000); camera.position.set(0, 0, 330);
    const world = new T.Group(); world.rotation.x = -0.34; scene.add(world);
    // faint sphere fill + rim
    world.add(new T.Mesh(new T.SphereGeometry(R * 0.985, 48, 36), new T.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.012 })));
    world.add(new T.Mesh(new T.SphereGeometry(R * 1.04, 40, 28), new T.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.02, side: T.BackSide })));
    // graticule
    const gp = [];
    for (let lat = -75; lat <= 75; lat += 15) { let pv = null; for (let lng = -180; lng <= 180; lng += 4) { const v = this.ll2v(lat, lng, R); if (pv) gp.push(pv.x, pv.y, pv.z, v.x, v.y, v.z); pv = v; } }
    for (let lng = -180; lng < 180; lng += 15) { let pv = null; for (let lat = -90; lat <= 90; lat += 4) { const v = this.ll2v(lat, lng, R); if (pv) gp.push(pv.x, pv.y, pv.z, v.x, v.y, v.z); pv = v; } }
    const gg = new T.BufferGeometry(); gg.setAttribute('position', new T.Float32BufferAttribute(gp, 3));
    world.add(new T.LineSegments(gg, new T.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.09 })));
    // coastlines (reuse cached land if present)
    this.addSplashCoast(world, R);
    this._splashGlobe = { renderer, scene, camera, world };
    const tick = () => { world.rotation.y += 0.0016; renderer.render(scene, camera); this._sraf = requestAnimationFrame(tick); };
    this._sraf = requestAnimationFrame(tick);
  }
  async addSplashCoast(world, R) {
    const T = window.THREE;
    try {
      await this.waitFor(() => window.topojson, 5000);
      if (!this._coastTopo) { const res = await fetch('https://unpkg.com/world-atlas@2.0.2/land-110m.json'); this._coastTopo = await res.json(); }
      const fc = window.topojson.feature(this._coastTopo, this._coastTopo.objects.land), pos = [];
      const addRing = ring => { let pv = null; for (const [lng, lat] of ring) { const v = this.ll2v(lat, lng, R * 1.001); if (pv) pos.push(pv.x, pv.y, pv.z, v.x, v.y, v.z); pv = v; } };
      for (const f of fc.features) { const gm = f.geometry; if (gm.type === 'Polygon') gm.coordinates.forEach(addRing); else if (gm.type === 'MultiPolygon') gm.coordinates.forEach(p => p.forEach(addRing)); }
      const g = new T.BufferGeometry(); g.setAttribute('position', new T.Float32BufferAttribute(pos, 3));
      world.add(new T.LineSegments(g, new T.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.16 })));
    } catch (e) {}
  }

  // config
  defaultConfig() {
    return {
      satName: 'LEOPT-1', opm: this.sampleOPM(), alt: 520, inc: 97.4, raan: 128, u0: 22,
      sigma0: 0.3, beamBeta: 0.34, holdNom: 45, mount: 'standard_dish', satAnt: 'turnstile',
      windowH: 6, growth: this.props.growthRate ?? 1.4, minEl: this.props.minElevation ?? 8,
      truthDeg: this.props.truthDeg ?? 1.8, strategy: this.props.strategy ?? 'auto',
      stations: [
        { name: 'SVALBARD', lat: 78.23, lng: 15.41 },
        { name: 'TROLL', lat: -72.01, lng: 2.53 },
        { name: 'FAIRBANKS', lat: 64.80, lng: -147.72 },
        { name: 'PUNTA ARENAS', lat: -53.00, lng: -70.85 },
        { name: 'DONGARA', lat: -29.05, lng: 115.35 },
        { name: 'HARTEBEESTHOEK', lat: -25.89, lng: 27.69 }
      ]
    };
  }
  presetStations(name) {
    if (name === 'south') return [
      { name: 'TROLL', lat: -72.01, lng: 2.53 }, { name: 'PUNTA ARENAS', lat: -53.00, lng: -70.85 },
      { name: 'DONGARA', lat: -29.05, lng: 115.35 }, { name: 'HARTEBEESTHOEK', lat: -25.89, lng: 27.69 },
      { name: 'AWARUA', lat: -46.53, lng: 168.38 }, { name: "O'HIGGINS", lat: -63.32, lng: -57.90 }
    ];
    if (name === 'min') return [
      { name: 'SVALBARD', lat: 78.23, lng: 15.41 }, { name: 'TROLL', lat: -72.01, lng: 2.53 }
    ];
    return this.defaultConfig().stations;
  }

  // scenario
  setup() {
    if (!this.config) this.config = this.defaultConfig();
    if (!this.form) this.form = JSON.parse(JSON.stringify(this.config));
    const c = this.config, mu = 398600.4418, a = 6371 + (+c.alt || 520);
    const period = 2 * Math.PI * Math.sqrt(a * a * a / mu);
    const P = this.P = {
      inc: +c.inc || 0, raan: +c.raan || 0, u0: +c.u0 || 0, period, we: 360 / 86164,
      Re: 6371, h: +c.alt || 520, window: Math.max(0.5, +c.windowH || 6) * 3600,
      minEl: Math.max(0, +c.minEl || 8), holdNom: Math.max(5, +c.holdNom || 45),
      beta: Math.max(0.05, +c.beamBeta || 0.34), growth: (Math.max(0, +c.growth || 1.4)) / 3600,
      dDelta: 0.05, nGrid: 241
    };
    this.alt = P.h / P.Re;
    this.stations = (c.stations || []).map(s => ({
      name: s.name, code: ((s.name || 'STN').replace(/[^A-Za-z0-9]/g, '').slice(0, 3).toUpperCase() || 'STN'),
      lat: +s.lat || 0, lng: +s.lng || 0
    }));
    this.covAng = this.solveCoverage();
    const s0 = Math.max(0.05, +c.sigma0 || 0.3);
    this.grid = new Float64Array(P.nGrid);
    this.deltas = new Float64Array(P.nGrid);
    for (let k = 0; k < P.nGrid; k++) { const d = -6 + k * P.dDelta; this.deltas[k] = d; this.grid[k] = Math.exp(-(d * d) / (2 * s0 * s0)); }
    this.normalize();
    this.tLast = 0; this.log = [];
    this.history = [{ t: 0, ent: this.entropy(this.grid) }];
    this.buildPasses(); this.buildSchedule();
    this.maxGap = this.computeMaxGap();
    this.computeGanttColors();
  }

  // math
  d2r(x){return x*Math.PI/180} r2d(x){return x*180/Math.PI}
  uNow(t){ return this.P.u0 + 360 * t / this.P.period; }
  subpoint(u, t) {
    const P = this.P, i = this.d2r(P.inc), ur = this.d2r(u);
    const lat = this.r2d(Math.asin(Math.sin(i) * Math.sin(ur)));
    let lon = P.raan + this.r2d(Math.atan2(Math.cos(i) * Math.sin(ur), Math.cos(ur)));
    lon -= P.we * t;
    lon = ((lon + 180) % 360 + 360) % 360 - 180;
    return { lat, lng: lon };
  }
  centralAngle(la1, lo1, la2, lo2) {
    const a = this.d2r(la1), b = this.d2r(la2), dl = this.d2r(lo2 - lo1);
    return this.r2d(Math.acos(Math.min(1, Math.max(-1, Math.sin(a)*Math.sin(b) + Math.cos(a)*Math.cos(b)*Math.cos(dl)))));
  }
  elevation(c) { const k = this.P.Re / (this.P.Re + this.P.h), cc = this.d2r(c); return this.r2d(Math.atan2(Math.cos(cc) - k, Math.sin(cc))); }
  solveCoverage() { for (let c = 0.5; c < 35; c += 0.1) if (this.elevation(c) < this.P.minEl) return c; return 20; }
  azimuth(la1, lo1, la2, lo2) {
    const a = this.d2r(la1), b = this.d2r(la2), dl = this.d2r(lo2 - lo1);
    return (this.r2d(Math.atan2(Math.sin(dl) * Math.cos(b), Math.cos(a) * Math.sin(b) - Math.sin(a) * Math.cos(b) * Math.cos(dl))) + 360) % 360;
  }
  ll2v(lat, lng, r) {
    const T = window.THREE, phi = (90 - lat) * Math.PI / 180, th = (lng + 90) * Math.PI / 180;
    return new T.Vector3(-r * Math.sin(phi) * Math.cos(th), r * Math.cos(phi), r * Math.sin(phi) * Math.sin(th));
  }

  // belief
  normalize() { let s = 0; for (const v of this.grid) s += v; if (s > 0) for (let i = 0; i < this.grid.length; i++) this.grid[i] /= s; }
  entropy(arr) { let s = 0; for (const p of arr) if (p > 1e-9) s -= p * Math.log2(p); return s; }
  gaussBlur(arr, sb) {
    if (sb < 0.4) return arr.slice(0);
    const r = Math.min(arr.length - 1, Math.ceil(sb * 3)), k = []; let ks = 0;
    for (let i = -r; i <= r; i++) { const v = Math.exp(-(i * i) / (2 * sb * sb)); k.push(v); ks += v; }
    const out = new Float64Array(arr.length);
    for (let i = 0; i < arr.length; i++) { let s = 0; for (let j = -r; j <= r; j++) { const idx = i + j; if (idx < 0 || idx >= arr.length) continue; s += arr[idx] * k[j + r]; } out[i] = s / ks; }
    return out;
  }
  displayGrid(t) {
    const sb = (this.P.growth * Math.max(0, t - this.tLast)) / this.P.dDelta;
    const g = this.gaussBlur(this.grid, sb);
    let s = 0; for (const v of g) s += v; if (s > 0) for (let i = 0; i < g.length; i++) g[i] /= s;
    return g;
  }
  pdet(d, aim, g) { const x = (d - aim) / this.P.beta; return g * Math.exp(-x * x / 2); }
  expectedDetect(aim, g, grid) { let s = 0; for (let k = 0; k < grid.length; k++) s += grid[k] * this.pdet(this.deltas[k], aim, g); return s; }
  geomQuality(pass) { return 0.55 + 0.4 * Math.min(1, pass.maxEl / 70); }
  planAim(slot, grid) {
    const g = this.geomQuality(slot.pass), b = this.P.beta;
    if (this.state.strategy === 'sweep') { let a = -4 + this.log.length * 2 * b; a = ((a + 5) % 10 + 10) % 10 - 5; return Math.round(a * 100) / 100; }
    let best = -1, bestA = 0;
    for (let a = -5; a <= 5; a += b) { const e = this.expectedDetect(a, g, grid); if (e > best) { best = e; bestA = a; } }
    return Math.round(bestA * 100) / 100;
  }

  // passes
  buildPasses() {
    const P = this.P; this.passesByStation = {}; this.allPasses = [];
    for (const st of this.stations) {
      const arr = []; let inPass = false, cur = null;
      for (let t = 0; t <= P.window; t += 15) {
        const sp = this.subpoint(this.uNow(t), t);
        const el = this.elevation(this.centralAngle(st.lat, st.lng, sp.lat, sp.lng));
        if (el >= P.minEl) {
          if (!inPass) { inPass = true; cur = { station: st, tAOS: t, tLOS: t, maxEl: el, tmax: t }; }
          else { cur.tLOS = t; if (el > cur.maxEl) { cur.maxEl = el; cur.tmax = t; } }
        } else if (inPass) { inPass = false; if (cur.tLOS - cur.tAOS >= 30) arr.push(cur); cur = null; }
      }
      if (inPass && cur && cur.tLOS - cur.tAOS >= 30) arr.push(cur);
      this.passesByStation[st.name] = arr;
      for (const p of arr) this.allPasses.push(p);
    }
  }
  buildSchedule() {
    const P = this.P, slots = [];
    for (const p of this.allPasses) {
      const dur = p.tLOS - p.tAOS, n = Math.max(1, Math.min(3, Math.floor(dur / P.holdNom)));
      p.nSlots = n;
      for (let i = 0; i < n; i++) { const frac = (i + 1) / (n + 1); slots.push({ station: p.station, pass: p, t: p.tAOS + frac * dur, hold: P.holdNom }); }
    }
    slots.sort((a, b) => a.t - b.t); this.schedule = slots;
  }
  computeMaxGap() {
    const iv = this.allPasses.map(p => [p.tAOS, p.tLOS]).sort((a, b) => a[0] - b[0]);
    let gap = 0, cursor = 0;
    for (const [a, b] of iv) { if (a > cursor) gap = Math.max(gap, a - cursor); cursor = Math.max(cursor, b); }
    return Math.max(gap, this.P.window - cursor);
  }
  computeGanttColors() {
    for (const p of this.allPasses) {
      const g = this.geomQuality(p), grid = this.displayGrid(Math.min(p.tmax, this.P.window));
      let best = 0; for (let a = -5; a <= 5; a += this.P.beta) { const e = this.expectedDetect(a, g, grid); if (e > best) best = e; }
      p.__pAcq = 1 - Math.pow(1 - best, p.nSlots || 1); p.__bg = this.pFill(p.__pAcq); p.__border = this.pStroke(p.__pAcq, 0.55);
    }
  }

  // colors
  pStroke(p, a) {
    p = Math.max(0, Math.min(1, p)); let x, y, f;
    if (p < 0.5) { x = [205, 99, 90]; y = [216, 168, 90]; f = p / 0.5; } else { x = [216, 168, 90]; y = [107, 191, 148]; f = (p - 0.5) / 0.5; }
    const c = x.map((v, i) => Math.round(v + (y[i] - v) * f));
    return a == null ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  }
  pFill(p) { return this.pStroke(p, 0.16 + 0.26 * Math.max(0, Math.min(1, p))); }
  beliefRGB(w) { const t = Math.max(0, Math.min(1, w)); return [(48 + 78 * t) / 255, (92 + 116 * t) / 255, (150 + 105 * t) / 255]; }
  makeSoftTex() {
    const T = window.THREE, c = document.createElement('canvas'); c.width = c.height = 64;
    const x = c.getContext('2d'), g = x.createRadialGradient(32, 32, 0, 32, 32, 32);
    g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(0.45, 'rgba(255,255,255,0.4)'); g.addColorStop(1, 'rgba(255,255,255,0)');
    x.fillStyle = g; x.fillRect(0, 0, 64, 64);
    const tex = new T.CanvasTexture(c); tex.needsUpdate = true; return tex;
  }
  hex(rgbStr) { const m = rgbStr.match(/\d+/g); return (+m[0] << 16) | (+m[1] << 8) | (+m[2]); }

  // three.js globe
  waitFor(cond, ms = 8000) { return new Promise(res => { const t0 = Date.now(); const i = setInterval(() => { if (cond() || Date.now() - t0 > ms) { clearInterval(i); res(); } }, 40); }); }
  async initGlobe() {
    await this.waitFor(() => window.THREE && this.globeEl && this.globeEl.clientWidth > 0);
    const T = window.THREE; if (!T || !this.globeEl) return;
    const el = this.globeEl, W = el.clientWidth, H = el.clientHeight, R = this.R = 100;
    const renderer = new T.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.setSize(W, H); renderer.setClearColor(0x000000, 0);
    renderer.domElement.style.display = 'block'; el.appendChild(renderer.domElement);
    const scene = new T.Scene();
    scene.add(new T.AmbientLight(0xffffff, 0.7)); const __dl = new T.DirectionalLight(0xffffff, 0.9); __dl.position.set(0.6, 0.8, 1); scene.add(__dl);
    const camera = new T.PerspectiveCamera(40, W / H, 1, 6000);
    this.camDist = 330; camera.position.set(0, 0, this.camDist);
    const world = new T.Group(); world.rotation.x = -0.32; world.rotation.y = -1.1; scene.add(world);
    this.gl = { T, renderer, scene, camera, world };
    world.add(new T.Mesh(new T.SphereGeometry(R, 64, 48), new T.MeshBasicMaterial({ color: 0x11151c })));
    world.add(new T.Mesh(new T.SphereGeometry(R * 1.08, 48, 32), new T.MeshBasicMaterial({ color: 0x7aa6f0, transparent: true, opacity: 0.05, side: T.BackSide })));
    const gp = [];
    for (let lat = -75; lat <= 75; lat += 15) { let pv = null; for (let lng = -180; lng <= 180; lng += 4) { const v = this.ll2v(lat, lng, R * 1.002); if (pv) gp.push(pv.x, pv.y, pv.z, v.x, v.y, v.z); pv = v; } }
    for (let lng = -180; lng < 180; lng += 15) { let pv = null; for (let lat = -88; lat <= 88; lat += 4) { const v = this.ll2v(lat, lng, R * 1.002); if (pv) gp.push(pv.x, pv.y, pv.z, v.x, v.y, v.z); pv = v; } }
    const gg = new T.BufferGeometry(); gg.setAttribute('position', new T.Float32BufferAttribute(gp, 3));
    world.add(new T.LineSegments(gg, new T.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.055 })));
    const mkLine = (col, op) => { const l = new T.Line(new T.BufferGeometry(), new T.LineBasicMaterial({ color: col, transparent: true, opacity: op })); l.frustumCulled = false; world.add(l); return l; };
    this.gl.orbit = mkLine(0x7aa6f0, 0.9); this.gl.gtrack = mkLine(0xffffff, 0.12);
    const softTex = this.makeSoftTex();
    this.gl.belief = new T.Points(new T.BufferGeometry(), new T.PointsMaterial({ size: 7.5, map: softTex, vertexColors: true, transparent: true, opacity: 0.5, sizeAttenuation: true, depthWrite: false }));
    this.gl.belief.frustumCulled = false; world.add(this.gl.belief);
    this.gl.uncArc = new T.Line(new T.BufferGeometry(), new T.LineBasicMaterial({ color: 0x7aa6f0, transparent: true, opacity: 0.3 })); this.gl.uncArc.frustumCulled = false; world.add(this.gl.uncArc);
    this.gl.stationGroup = new T.Group(); world.add(this.gl.stationGroup); this.buildStationMeshes();
    this.gl.beam = new T.Mesh(new T.CircleGeometry(1, 44), new T.MeshBasicMaterial({ color: 0xe0a96b, transparent: true, opacity: 0.3, side: T.DoubleSide, depthWrite: false })); this.gl.beam.frustumCulled = false; world.add(this.gl.beam);
    this.gl.beamRing = new T.LineLoop(new T.BufferGeometry(), new T.LineBasicMaterial({ color: 0xe0a96b })); this.gl.beamRing.frustumCulled = false; world.add(this.gl.beamRing);
    this.gl.los = mkLine(0xe0a96b, 0.65);
    this.gl.pulse = new T.Mesh(new T.RingGeometry(0.92, 1, 48), new T.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5, side: T.DoubleSide, depthWrite: false })); world.add(this.gl.pulse);
    this.gl.sat = this.buildSatModel(); this.gl.sat.frustumCulled = false; world.add(this.gl.sat);
    this.gl.truth = new T.Mesh(new T.SphereGeometry(1.5, 10, 10), new T.MeshBasicMaterial({ color: 0xd8786e })); this.gl.truth.frustumCulled = false; world.add(this.gl.truth);
    this.labelLayer = document.createElement('div'); this.labelLayer.style.cssText = 'position:absolute;inset:0;pointer-events:none;overflow:hidden'; el.appendChild(this.labelLayer);
    this.buildLabels();
    this.loadCoast(); this.bindControls();
    this._globeReady = true; this.updateGlobe(true); this.startLoop();
    this._ro = new ResizeObserver(() => { const w2 = el.clientWidth, h2 = el.clientHeight; renderer.setSize(w2, h2); camera.aspect = w2 / h2; camera.updateProjectionMatrix(); }); this._ro.observe(el);
  }
  buildStationMeshes() {
    const T = this.gl.T, R = this.R, grp = this.gl.stationGroup;
    while (grp.children.length) { const c = grp.children.pop(); if (c.geometry) c.geometry.dispose(); grp.remove(c); }
    for (const st of this.stations) {
      const m = new T.Mesh(new T.SphereGeometry(1.1, 10, 10), new T.MeshBasicMaterial({ color: 0xeef0f3 })); m.position.copy(this.ll2v(st.lat, st.lng, R * 1.004)); grp.add(m);
      const cg = new T.BufferGeometry(); cg.setAttribute('position', new T.Float32BufferAttribute(this.circlePts(st.lat, st.lng, this.covAng, R * 1.004), 3));
      grp.add(new T.LineLoop(cg, new T.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.16 })));
    }
  }
  rebuildStations() { if (!this._globeReady) return; this.buildStationMeshes(); while (this.labelLayer.firstChild) this.labelLayer.removeChild(this.labelLayer.firstChild); this.buildLabels(); }
  circlePts(lat, lng, radDeg, r, n = 64) {
    const out = [], la = this.d2r(lat), lo = this.d2r(lng), dr = this.d2r(radDeg);
    for (let i = 0; i <= n; i++) {
      const br = this.d2r(i / n * 360);
      const la2 = Math.asin(Math.sin(la) * Math.cos(dr) + Math.cos(la) * Math.sin(dr) * Math.cos(br));
      const lo2 = lo + Math.atan2(Math.sin(br) * Math.sin(dr) * Math.cos(la), Math.cos(dr) - Math.sin(la) * Math.sin(la2));
      const v = this.ll2v(this.r2d(la2), this.r2d(lo2), r); out.push(v.x, v.y, v.z);
    }
    return out;
  }
  buildLabels() {
    this.labels = [];
    const mk = html => { const d = document.createElement('div'); d.className = 'gl-html'; d.innerHTML = html; this.labelLayer.appendChild(d); return d; };
    for (const st of this.stations) {
      const el = mk(`<div style="display:flex;align-items:center;gap:5px"><span style="width:5px;height:5px;background:#eef0f3;display:inline-block;border-radius:50%"></span><span style="font-size:9px;color:#c3c8d0;letter-spacing:.04em;text-shadow:0 1px 3px #000">${st.code}</span></div>`);
      const v = this.ll2v(st.lat, st.lng, this.R * 1.03); this.labels.push({ el, src: () => v });
    }
    this.labels.push({ el: mk('<span style="font-size:8px;color:#cfe0ff;letter-spacing:.06em;text-shadow:0 1px 3px #000">&#9670; SAT</span>'), src: () => this.gl.sat.position });
    this.labels.push({ el: mk('<span style="font-size:8px;color:#e6a79f;letter-spacing:.06em;text-shadow:0 1px 3px #000">&#9671; TRUTH</span>'), src: () => this.gl.truth.position, hidden: () => !this.state.revealed });
  }
  async loadCoast() {
    try {
      await this.waitFor(() => window.topojson, 5000);
      const res = await fetch('https://unpkg.com/world-atlas@2.0.2/land-110m.json');
      const topo = await res.json(); const fc = window.topojson.feature(topo, topo.objects.land);
      const T = window.THREE, pos = [];
      const addRing = ring => { let pv = null; for (const [lng, lat] of ring) { const v = this.ll2v(lat, lng, this.R * 1.005); if (pv) pos.push(pv.x, pv.y, pv.z, v.x, v.y, v.z); pv = v; } };
      for (const f of fc.features) { const gm = f.geometry; if (gm.type === 'Polygon') gm.coordinates.forEach(addRing); else if (gm.type === 'MultiPolygon') gm.coordinates.forEach(p => p.forEach(addRing)); }
      const g = new T.BufferGeometry(); g.setAttribute('position', new T.Float32BufferAttribute(pos, 3));
      this.gl.world.add(new T.LineSegments(g, new T.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.2 })));
    } catch (e) {}
  }
  bindControls() {
    const el = this.gl.renderer.domElement, T = this.gl.T; let drag = false, px = 0, py = 0; el.style.cursor = 'grab';
    el.addEventListener('pointerdown', e => { drag = true; this._dragging = true; px = e.clientX; py = e.clientY; this._userInteract = true; try { el.setPointerCapture(e.pointerId); } catch (x) {} el.style.cursor = 'grabbing'; if (this.tip) this.tip.style.display = 'none'; });
    el.addEventListener('pointermove', e => { if (drag) { const dx = e.clientX - px, dy = e.clientY - py; px = e.clientX; py = e.clientY; const w = this.gl.world; w.rotation.y += dx * 0.005; w.rotation.x = Math.max(-1.3, Math.min(1.3, w.rotation.x + dy * 0.005)); } else { this.hover(e); } });
    const up = () => { drag = false; this._dragging = false; el.style.cursor = 'grab'; }; el.addEventListener('pointerup', up);
    el.addEventListener('pointerleave', () => { drag = false; this._dragging = false; el.style.cursor = 'grab'; if (this.tip) this.tip.style.display = 'none'; });
    el.addEventListener('wheel', e => { e.preventDefault(); this.camDist = Math.max(150, Math.min(700, this.camDist + e.deltaY * 0.25)); }, { passive: false });
    this.ray = new T.Raycaster(); this.ray.params.Points.threshold = 8;
    this.tip = document.createElement('div'); this.tip.style.cssText = "position:absolute;pointer-events:none;display:none;z-index:8;min-width:188px;background:rgba(13,15,19,.93);border:1px solid rgba(122,166,240,.45);border-radius:7px;padding:10px 12px;backdrop-filter:blur(4px);box-shadow:0 6px 22px rgba(0,0,0,.45)"; this.globeEl.appendChild(this.tip);
  }
  hover(e) {
    if (!this._globeReady || !this.tip || this._dragging) return;
    const el = this.globeEl, r = el.getBoundingClientRect();
    const mx = ((e.clientX - r.left) / r.width) * 2 - 1, my = -((e.clientY - r.top) / r.height) * 2 + 1;
    this.ray.setFromCamera({ x: mx, y: my }, this.gl.camera);
    const hit = this.ray.intersectObject(this.gl.sat, true).length > 0 || this.ray.intersectObject(this.gl.belief, false).length > 0;
    if (!hit) { this.tip.style.display = 'none'; el.style.cursor = 'grab'; return; }
    const P = this.P, disp = this.displayGrid(this.state.t);
    let mean = 0; for (let k = 0; k < P.nGrid; k++) mean += disp[k] * this.deltas[k];
    let v = 0; for (let k = 0; k < P.nGrid; k++) v += disp[k] * (this.deltas[k] - mean) ** 2; const sig = Math.sqrt(v);
    const ent = this.entropy(disp), km = Math.round(3 * sig * Math.PI / 180 * (P.Re + P.h));
    const row = (k, val, c) => `<div style="display:flex;justify-content:space-between;gap:20px;font-size:10px;line-height:1.8"><span style="color:#8b929c">${k}</span><span style="color:${c || '#e6e8ec'};font-family:'IBM Plex Mono',monospace">${val}</span></div>`;
    this.tip.innerHTML = `<div style="color:#cfe0ff;font-size:10px;font-weight:600;letter-spacing:.08em;margin-bottom:6px">POSITION UNCERTAINTY</div>` +
      row('Along-track 1\u03c3', sig.toFixed(2) + '\u00b0', '#e0a96b') +
      row('3\u03c3 search span', '\u00b1' + (3 * sig).toFixed(2) + '\u00b0 \u00b7 ' + km + ' km') +
      row('Est. offset', (mean >= 0 ? '+' : '') + mean.toFixed(2) + '\u00b0') +
      row('Belief entropy', ent.toFixed(2) + ' bits') +
      row('Status', this.state.acquired ? 'ACQUIRED' : 'SEARCHING', this.state.acquired ? '#6bbf94' : '#e0a96b');
    let x = e.clientX - r.left + 16, y = e.clientY - r.top + 16;
    if (x > r.width - 210) x = e.clientX - r.left - 204; if (y > r.height - 130) y = e.clientY - r.top - 130;
    this.tip.style.left = x + 'px'; this.tip.style.top = y + 'px'; this.tip.style.display = 'block'; el.style.cursor = 'help';
  }
  setLine(line, arr) { const T = this.gl.T, g = line.geometry; g.setAttribute('position', new T.Float32BufferAttribute(arr, 3)); g.attributes.position.needsUpdate = true; }
  placeDisc(mesh, lat, lng, radDeg, r) { const T = this.gl.T, p = this.ll2v(lat, lng, r); mesh.position.copy(p); mesh.quaternion.setFromUnitVectors(new T.Vector3(0, 0, 1), p.clone().normalize()); mesh.scale.setScalar(Math.max(0.6, this.R * Math.sin(this.d2r(radDeg)))); }
  startLoop() {
    const tick = () => {
      if (!this._globeReady) return; const g = this.gl;
      try {
        if (!this._userInteract) g.world.rotation.y += 0.0009;
        g.camera.position.z = this.camDist; g.world.updateMatrixWorld(); if (g.sat) g.sat.rotation.y += 0.012;
        if (g.pulse.visible) { this._pt = (this._pt || 0) + 0.04; const ph = (Math.sin(this._pt) + 1) / 2; g.pulse.scale.setScalar(this.R * Math.sin(this.d2r(this.covAng)) * (0.35 + 0.65 * ph)); g.pulse.material.opacity = 0.5 * (1 - ph) + 0.1; }
        g.renderer.render(g.scene, g.camera);
        this.updateLabels();
      } catch (e) { window.__loopErr = String(e && e.stack || e); }
      this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }
  updateLabels() {
    if (!this.labels) return; const g = this.gl, W = this.globeEl.clientWidth, H = this.globeEl.clientHeight, mw = g.world.matrixWorld;
    for (const lb of this.labels) {
      if (lb.hidden && lb.hidden()) { lb.el.style.display = 'none'; continue; }
      const p = lb.src().clone().applyMatrix4(mw);
      if (p.z < 4) { lb.el.style.display = 'none'; continue; }
      const pr = p.clone().project(g.camera); if (pr.z > 1) { lb.el.style.display = 'none'; continue; }
      const x = (pr.x * 0.5 + 0.5) * W, y = (-pr.y * 0.5 + 0.5) * H;
      lb.el.style.display = ''; lb.el.style.transform = `translate(-50%,-50%) translate(${x}px,${y}px)`;
    }
  }
  updateGlobe(full) { if (!this._globeReady || !this.gl) return; try { this._upd(full); this.gl.renderer.render(this.gl.scene, this.gl.camera); } catch (e) { window.__leoptErr = String(e && e.stack || e); console.error('updateGlobe', e); } }
  _upd(full) {
    const T = this.gl.T, R = this.R, P = this.P, t = this.state.t, disp = this.displayGrid(t);
    const span = P.period * 0.62, op = [], gpa = [];
    for (let s = -span; s <= span; s += 18) { const tt = t + s; if (tt < 0 || tt > P.window) continue; const sp = this.subpoint(this.uNow(tt), tt); const a = this.ll2v(sp.lat, sp.lng, R * (1 + this.alt)); op.push(a.x, a.y, a.z); const b = this.ll2v(sp.lat, sp.lng, R * 1.002); gpa.push(b.x, b.y, b.z); }
    this.setLine(this.gl.orbit, op); this.setLine(this.gl.gtrack, gpa);
    const maxW = Math.max(...disp) || 1, bpos = [], bcol = [], arc = [];
    let bmean = 0; for (let k = 0; k < P.nGrid; k++) bmean += disp[k] * this.deltas[k];
    for (let k = 0; k < P.nGrid; k += 2) {
      const wgt = disp[k] / maxW; const sp = this.subpoint(this.uNow(t) + this.deltas[k], t); const v = this.ll2v(sp.lat, sp.lng, R * (1 + this.alt));
      if (wgt >= 0.02) arc.push(v.x, v.y, v.z);
      if (wgt < 0.04) continue;
      bpos.push(v.x, v.y, v.z); const c = this.beliefRGB(wgt); bcol.push(c[0], c[1], c[2]);
    }
    const bg = this.gl.belief.geometry; bg.setAttribute('position', new T.Float32BufferAttribute(bpos, 3)); bg.setAttribute('color', new T.Float32BufferAttribute(bcol, 3)); bg.computeBoundingSphere();
    this.setLine(this.gl.uncArc, arc);
    const spN = this.subpoint(this.uNow(t) + bmean, t); this.gl.sat.position.copy(this.ll2v(spN.lat, spN.lng, R * (1 + this.alt)));
    if (this.state.revealed) { const spT = this.subpoint(this.uNow(t) + this.state.truthDeg, t); this.gl.truth.position.copy(this.ll2v(spT.lat, spT.lng, R * (1 + this.alt))); this.gl.truth.visible = true; } else this.gl.truth.visible = false;
    if (!full) return;
    const slot = this.schedule[this.state.stepIndex];
    if (slot && !this.state.acquired) {
      const aim = this.planAim(slot, this.displayGrid(slot.t)), g = this.geomQuality(slot.pass), pd = this.expectedDetect(aim, g, this.displayGrid(slot.t));
      const spA = this.subpoint(this.uNow(slot.t) + aim, slot.t), col = this.hex(this.pStroke(pd));
      this.placeDisc(this.gl.beam, spA.lat, spA.lng, 1.6, R * 1.006); this.gl.beam.material.color.setHex(col); this.gl.beam.material.opacity = 0.16 + 0.46 * pd; this.gl.beam.visible = true;
      this.setLine(this.gl.beamRing, this.circlePts(spA.lat, spA.lng, 1.6, R * 1.007)); this.gl.beamRing.material.color.setHex(col); this.gl.beamRing.visible = true;
      const a = this.ll2v(slot.station.lat, slot.station.lng, R * 1.004), b = this.ll2v(spA.lat, spA.lng, R * (1 + this.alt));
      this.setLine(this.gl.los, [a.x, a.y, a.z, b.x, b.y, b.z]); this.gl.los.material.color.setHex(col); this.gl.los.visible = true;
      this.placeDisc(this.gl.pulse, slot.station.lat, slot.station.lng, this.covAng, R * 1.004); this.gl.pulse.visible = true;
    } else { this.gl.beam.visible = this.gl.beamRing.visible = this.gl.los.visible = this.gl.pulse.visible = false; }
  }
  treatStyle(w, idx) {
    const tr = this.state.treatment;
    if (tr === 'tube') return { alt: this.alt * 0.5 };
    if (tr === 'swarm') return { alt: 0.004 + (Math.sin(idx * 12.9898) * 0.5 + 0.5) * 0.05 };
    return { alt: 0.006 };
  }

  // actions
  togglePlay() {
    if (this.state.playing) { clearInterval(this._timer); this.setState({ playing: false }); return; }
    clearInterval(this._auto); this.setState({ playing: true, mode: 'rehearse' });
    this._timer = setInterval(() => {
      let nt = this.state.t + 75;
      if (nt >= this.P.window) { clearInterval(this._timer); this.setState({ t: this.P.window, playing: false }, () => this.updateGlobe(false)); return; }
      this.setState({ t: nt, tickN: this.state.tickN + 1 }, () => this.updateGlobe(false));
    }, 90);
  }
  scrub(e) { const v = +e.target.value; clearInterval(this._timer); this.setState({ t: v, playing: false }, () => this.updateGlobe(true)); }
  setTruth(e) { this.setState({ truthDeg: +e.target.value }, () => this.updateGlobe(true)); }
  toggleReveal() { this.setState({ revealed: !this.state.revealed }, () => this.updateGlobe(false)); }
  setStrategy(s) { this.setState({ strategy: s }, () => { this.computeGanttColors(); this.updateGlobe(true); }); }
  // Header controller badge: prefer the backend's real active controller (from a
  // live plan), else the locally selected strategy.
  strategyBadge(st, ctl) {
    if (ctl && ctl.controller_name) {
      return { strategyLabel: String(ctl.controller_name).toUpperCase(), strategyColor: ctl.is_agile ? '#7aa6f0' : '#c79be0' };
    }
    const m = { auto: ['AUTO-SELECT', '#7aa6f0'], infogreedy: ['INFO-GREEDY', '#7aa6f0'], pft: ['PFT-DPW PLANNER', '#c79be0'], pomcp: ['POMCP', '#7aa6f0'], sweep: ['NAÏVE SWEEP', '#e0a96b'] };
    const [label, color] = m[st.strategy] || ['CONTROLLER', '#7aa6f0'];
    return { strategyLabel: label, strategyColor: color };
  }
  _clearScheduleExportPatch() {
    this._scheduleExportToken = (this._scheduleExportToken || 0) + 1;
    this._scheduleExportActive = false;
    return { scheduleExportBusy: false, scheduleExportStatus: '', scheduleExportError: false };
  }
  // Mount picker: set the ground-antenna agility and preview the controller the
  // backend would auto-select for it.
  setMount(id) { if (!this.form) return; this.form.mount = id; this.setState({ ...this._clearScheduleExportPatch(), tickN: this.state.tickN + 1 }); this.previewController(); }
  async previewController() {
    if (!window.LeoptAPI || !window.LeoptLive || !this.form) return;
    try {
      const rec = await window.LeoptLive.recommendController(this.form);
      this._rec = rec; this.setState({ tickN: this.state.tickN + 1 });
    } catch (e) { /* preview is best-effort; ignore when the API is unavailable */ }
  }
  setTreatment(tr) { this.setState({ treatment: tr }, () => this.updateGlobe(true)); }
  setMode(m) { if (m === 'plan') this.form = JSON.parse(JSON.stringify(this.config || this.defaultConfig())); this.setState({ mode: m }); }
  launch() { clearInterval(this._timer); clearInterval(this._auto); this._auto = null; this.form = JSON.parse(JSON.stringify(this.config || this.defaultConfig())); this.setState({ launching: true, mode: 'plan' }); setTimeout(() => this.setState({ splashVisible: false, launching: false }), 620); }
  async _loadOrbitFile(file) {
    if (!file) return;
    try {
      const txt = await file.text();
      if (txt && txt.trim()) { this.setOpm(txt); this.setState({ tickN: this.state.tickN + 1 }); }
    } catch (x) { console.warn('[leopt] could not read orbit file', x); }
  }
  async readDroppedFile(e) {
    e.preventDefault();
    this.setState({ dragOver: false });
    const dt = e.dataTransfer; if (!dt) return;
    let file = dt.files && dt.files[0];
    if (!file && dt.items) { for (const it of dt.items) { if (it.kind === 'file') { file = it.getAsFile(); break; } } }
    this._loadOrbitFile(file);
  }
  readPickedFile(e) {
    const inp = e && e.target; const file = inp && inp.files && inp.files[0];
    this._loadOrbitFile(file);
    if (inp) inp.value = ''; // allow re-picking the same file
  }
  openFilePicker() { if (this.fileInput) this.fileInput.click(); }

  // planner
  setField(key, val) { if (!this.form) return; this.form[key] = val; this.setState({ ...this._clearScheduleExportPatch(), tickN: this.state.tickN + 1 }); }
  setStationField(i, key, val) { if (!this.form) return; this.form.stations[i][key] = val; this.setState({ ...this._clearScheduleExportPatch(), tickN: this.state.tickN + 1 }); }
  removeStation(i) { if (!this.form) return; this.form.stations.splice(i, 1); this.setState({ ...this._clearScheduleExportPatch(), tickN: this.state.tickN + 1 }); }
  addStation() { if (!this.form) return; this.form.stations.push({ name: 'NEW STATION', lat: 0, lng: 0 }); this.setState({ ...this._clearScheduleExportPatch(), tickN: this.state.tickN + 1 }); }
  applyPreset(name) { if (!this.form) return; this.form.stations = JSON.parse(JSON.stringify(this.presetStations(name))); this.setState({ ...this._clearScheduleExportPatch(), tickN: this.state.tickN + 1 }); }
  sampleOPM() { return 'LEOPT-1\n1 99999U 24001A   24001.00000000  .00000000  00000-0  00000-0 0  0001\n2 99999  97.4000 128.0000 0001000  90.0000  22.0000 15.21000000000010'; }
  // Auto-detect and parse a TLE or a CCSDS OPM (KVN) — either a Keplerian
  // element set or a state vector (X/Y/Z + velocities) — into the fields the
  // globe preview needs (inc, raan, alt, u0=argument-of-latitude, ecc). The real
  // ingest still happens server-side via /api/ingest; this drives the preview
  // and the "detected format" label.
  parseOrbit(txt) {
    txt = txt || ''; const out = {}; const lines = txt.split(/\r?\n/);
    const norm = a => ((a % 360) + 360) % 360;
    const l1 = lines.find(l => /^\s*1\s/.test(l)), l2 = lines.find(l => /^\s*2\s/.test(l));
    if (l1 && l2) {
      // TLE (line 2: inc raan ecc argp meanAnom meanMotion)
      const tk = l2.trim().split(/\s+/);
      const inc = parseFloat(tk[2]), raan = parseFloat(tk[3]);
      const ecc = parseFloat('0.' + (tk[4] || '0')), argp = parseFloat(tk[5]), ma = parseFloat(tk[6]), mm = parseFloat(tk[7]);
      if (isFinite(inc)) out.inc = inc; if (isFinite(raan)) out.raan = raan; if (isFinite(ecc)) out.ecc = ecc;
      if (isFinite(argp) && isFinite(ma)) out.u0 = norm(argp + ma); else if (isFinite(ma)) out.u0 = norm(ma);
      if (isFinite(mm) && mm > 0) { const T = 86400 / mm, a = Math.cbrt(398600.4418 * (T / (2 * Math.PI)) ** 2); out.alt = Math.round(a - 6371); }
      const l0 = lines.find(l => l.trim() && !/^\s*[12]\s/.test(l)); if (l0) out.satName = l0.trim();
      out.format = 'tle'; out.formatLabel = 'TLE';
      return out;
    }
    if (/=/.test(txt)) {
      // CCSDS OPM (KVN)
      const g = re => { const m = txt.match(re); return m ? parseFloat(m[1]) : NaN; };
      const nm = txt.match(/OBJECT_NAME\s*=\s*(.+)/i); if (nm) out.satName = nm[1].trim();
      const NUM = '([\\-\\d.eE+]+)';
      const X = g(new RegExp('(?:^|\\n)\\s*X\\s*=\\s*' + NUM)), Y = g(new RegExp('(?:^|\\n)\\s*Y\\s*=\\s*' + NUM)), Z = g(new RegExp('(?:^|\\n)\\s*Z\\s*=\\s*' + NUM));
      const VX = g(/X_DOT\s*=\s*([\-\d.eE+]+)/), VY = g(/Y_DOT\s*=\s*([\-\d.eE+]+)/), VZ = g(/Z_DOT\s*=\s*([\-\d.eE+]+)/);
      if ([X, Y, Z, VX, VY, VZ].every(Number.isFinite)) {
        // State vector. OPM positions are km/km·s^-1 by convention; fall back to
        // metres if the magnitude looks like SI.
        let r = [X, Y, Z], v = [VX, VY, VZ];
        if (Math.hypot(r[0], r[1], r[2]) > 1e5) { r = r.map(x => x / 1000); v = v.map(x => x / 1000); }
        const el = this._rv2elem(r, v);
        if (el) {
          if (isFinite(el.inc)) out.inc = el.inc; if (isFinite(el.raan)) out.raan = el.raan;
          if (isFinite(el.u)) out.u0 = norm(el.u); if (isFinite(el.a)) out.alt = Math.round(el.a - 6371);
          if (isFinite(el.ecc)) out.ecc = el.ecc;
        }
        out.format = 'opm'; out.formatLabel = 'OPM · state vector';
        return out;
      }
      // Keplerian element set.
      const inc = g(/INCLINATION\s*=\s*([\-\d.eE+]+)/i), raan = g(/RA_OF_ASC_NODE\s*=\s*([\-\d.eE+]+)/i),
        sma = g(/SEMI_MAJOR_AXIS\s*=\s*([\-\d.eE+]+)/i), argp = g(/ARG_OF_PERICENTER\s*=\s*([\-\d.eE+]+)/i),
        nu = g(/TRUE_ANOMALY\s*=\s*([\-\d.eE+]+)/i), ma = g(/MEAN_ANOMALY\s*=\s*([\-\d.eE+]+)/i), ecc = g(/ECCENTRICITY\s*=\s*([\-\d.eE+]+)/i);
      if (isFinite(inc)) out.inc = inc; if (isFinite(raan)) out.raan = raan;
      if (isFinite(sma)) out.alt = Math.round(sma - 6371); if (isFinite(ecc)) out.ecc = ecc;
      const anom = isFinite(nu) ? nu : ma;
      if (isFinite(argp) && isFinite(anom)) out.u0 = norm(argp + anom); else if (isFinite(anom)) out.u0 = norm(anom);
      if (out.inc != null || out.alt != null || out.raan != null) { out.format = 'opm'; out.formatLabel = 'OPM · Keplerian'; }
      return out;
    }
    return out;
  }
  // Classical elements from an ECI state vector (km, km/s). `u` is the argument
  // of latitude, which stays well-defined as eccentricity -> 0 (LEO is ~circular).
  _rv2elem(r, v) {
    const mu = 398600.4418, R2D = 180 / Math.PI;
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2], mag = a => Math.hypot(a[0], a[1], a[2]);
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const rm = mag(r), vm = mag(v); if (!(rm > 0) || !(vm > 0)) return null;
    const h = cross(r, v), hm = mag(h); const n = [-h[1], h[0], 0], nm = Math.hypot(n[0], n[1]);
    const energy = vm * vm / 2 - mu / rm; const a = energy !== 0 ? -mu / (2 * energy) : NaN;
    const eVec = r.map((ri, i) => ((vm * vm - mu / rm) * ri - dot(r, v) * v[i]) / mu), ecc = mag(eVec);
    const clamp = x => Math.max(-1, Math.min(1, x));
    const inc = Math.acos(clamp(h[2] / hm)) * R2D;
    let raan = nm > 1e-9 ? Math.acos(clamp(n[0] / nm)) * R2D : 0; if (n[1] < 0) raan = 360 - raan;
    let u = nm > 1e-9 ? Math.acos(clamp(dot(n, r) / (nm * rm))) * R2D : 0; if (r[2] < 0) u = 360 - u;
    return { a, inc, raan, u, ecc };
  }
  setOpm(val) { if (!this.form) return; this.form.opm = val; Object.assign(this.form, this.parseOrbit(val)); this.setState({ ...this._clearScheduleExportPatch(), tickN: this.state.tickN + 1 }); }
  loadSampleOrbit() { this.setOpm(this.sampleOPM()); }
  toggleAdv() { this.setState({ advOpen: !this.state.advOpen }); }
  setAdvTab(t) { this.setState({ advTab: t }); if (t === 'antenna' || t === 'satellite') this.previewController(); }
  advTabStyle(active) { return `padding:8px 13px;font-family:'IBM Plex Sans',sans-serif;font-size:9px;font-weight:600;letter-spacing:.12em;cursor:pointer;border:0;border-bottom:2px solid ${active ? '#7aa6f0' : 'transparent'};background:transparent;color:${active ? '#dce6fb' : '#6b727d'}`; }
  buildSatModel() {
    const T = this.gl.T, grp = new T.Group();
    grp.add(new T.Mesh(new T.BoxGeometry(1.6, 1.6, 2.4), new T.MeshStandardMaterial({ color: 0xeef2f7, metalness: 0.55, roughness: 0.4, emissive: 0x2a3340, emissiveIntensity: 0.5 })));
    const pm = new T.MeshStandardMaterial({ color: 0x29487f, metalness: 0.3, roughness: 0.5, emissive: 0x14305f, emissiveIntensity: 0.5 }), pg = new T.BoxGeometry(4.4, 0.1, 1.9);
    const pL = new T.Mesh(pg, pm); pL.position.x = -3.4; grp.add(pL);
    const pR = new T.Mesh(pg, pm); pR.position.x = 3.4; grp.add(pR);
    const dish = new T.Mesh(new T.CylinderGeometry(0.12, 0.7, 0.5, 14, 1, true), new T.MeshStandardMaterial({ color: 0xc7cdd6, metalness: 0.4, roughness: 0.5, side: T.DoubleSide })); dish.rotation.x = Math.PI / 2; dish.position.z = 1.55; grp.add(dish);
    grp.scale.setScalar(1.85); return grp;
  }
  rollout() {
    this.config = JSON.parse(JSON.stringify(this.form));
    this._livePlanGeneration = (this._livePlanGeneration || 0) + 1;
    this._livePlanPromise = null;
    this._liveMutationPromise = null;
    this.live = null; this.liveSessionId = null; this._liveCursor = 0; this._liveDwells = [];
    clearInterval(this._timer); clearInterval(this._auto); this._auto = null;
    this.setState({
      mode: 'rehearse', t: 0, stepIndex: 0, acquired: false, acqT: null, revealed: false, playing: false,
      strategy: this.config.strategy || 'auto', truthDeg: Math.max(-4, Math.min(4, +this.config.truthDeg || 0)), rolledOut: true,
      ...this._clearScheduleExportPatch(), tickN: this.state.tickN + 1
    }, () => { this.setup(); this.rebuildStations(); this.updateGlobe(true); this.forceUpdate(); this._goLive(); });
  }

  // Live engine bridge
  // LEOPT_LIVE enables backend observations. Export always uses the backend
  // session; globe geometry comes from the local Keplerian model.
  async _goLive(forceBackend) {
    if (this._livePlanPromise) return this._livePlanPromise;
    if (!window.LeoptLive) {
      if (forceBackend) throw new Error('server planning client is unavailable');
      return null;
    }
    if (!forceBackend && !window.LEOPT_LIVE) return null;
    const generation = this._livePlanGeneration || 0;
    this.live = null; this.liveSessionId = null; this._liveCursor = 0; this._liveDwells = [];
    const task = window.LeoptLive.planFromConfig(this.config, this.state);
    this._livePlanPromise = task;
    try {
      const live = await task;
      if (generation !== (this._livePlanGeneration || 0)) return null;
      this.live = live; this.liveSessionId = live.sessionId;
      // Flatten dwells in time order; the rehearse loop steps through these.
      this._liveDwells = [];
      for (const p of live.passes) for (const d of p.dwells) this._liveDwells.push({ passIdx: p.idx, dwellIdx: d.idx, ...d });
      console.info('[leopt-live] plan ok — session', live.sessionId,
        '| passes', live.passes.length, '| dwells', this._liveDwells.length,
        '| entropy', live.entropy != null ? live.entropy.toFixed(3) : '?');
      this.setState({ tickN: this.state.tickN + 1 });
      return live;
    } catch (e) {
      console.warn('[leopt-live] plan failed, staying on simulation:', e.message);
      if (forceBackend) throw e;
      return null;
    } finally {
      if (this._livePlanPromise === task) this._livePlanPromise = null;
    }
  }

  _decodeSchedulePdf(encoded) {
    if (typeof encoded !== 'string' || !encoded) throw new Error('server returned no PDF data');
    let raw;
    try { raw = window.atob(encoded); }
    catch (_e) { throw new Error('server returned invalid PDF encoding'); }
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    if (bytes.length < 5 || String.fromCharCode(...bytes.slice(0, 5)) !== '%PDF-') {
      throw new Error('server returned an invalid PDF file');
    }
    return bytes;
  }

  async _verifyScheduleDigest(bytes, expected, label) {
    if (!/^[0-9a-f]{64}$/i.test(String(expected || ''))) {
      throw new Error(`${label} checksum is missing or malformed`);
    }
    if (!window.crypto || !window.crypto.subtle) return;
    const view = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const digest = new Uint8Array(await window.crypto.subtle.digest('SHA-256', view));
    const actual = Array.from(digest, b => b.toString(16).padStart(2, '0')).join('');
    if (actual !== String(expected).toLowerCase()) {
      throw new Error(`${label} checksum verification failed`);
    }
  }

  _downloadScheduleBlob(blob, filename, fallback) {
    const safe = String(filename || fallback)
      .replace(/[\\/\u0000-\u001f\u007f]/g, '_')
      .slice(0, 180) || fallback;
    const url = URL.createObjectURL(blob), link = document.createElement('a');
    link.href = url; link.download = safe; link.style.display = 'none';
    document.body.appendChild(link); link.click(); link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async exportServerSchedule() {
    if (this._scheduleExportActive || this.state.scheduleExportBusy) return;
    this._scheduleExportActive = true;
    const token = (this._scheduleExportToken || 0) + 1;
    this._scheduleExportToken = token;
    this.setState({
      scheduleExportBusy: true,
      scheduleExportStatus: 'PREPARING SERVER PLAN…',
      scheduleExportError: false
    });
    try {
      if (this._liveMutationPromise) {
        this.setState({ scheduleExportStatus: 'WAITING FOR SERVER RE-PLAN…' });
        await this._liveMutationPromise;
        if (token !== this._scheduleExportToken) return;
      }
      let live = this.liveSessionId ? this.live : null;
      if (!live || !this.liveSessionId) live = await this._goLive(true);
      if (token !== this._scheduleExportToken) return;
      const sessionId = this.liveSessionId || (live && live.sessionId);
      if (!sessionId) throw new Error('the server did not create a plan session');
      if (!window.LeoptAPI || !window.LeoptAPI.exportSchedule) {
        throw new Error('schedule export API is unavailable');
      }
      this.setState({ scheduleExportStatus: 'RENDERING SERVER JSON + PDF…' });
      const bundle = await window.LeoptAPI.exportSchedule({ session_id: sessionId });
      if (token !== this._scheduleExportToken) return;
      if (!bundle || typeof bundle.json_content !== 'string' || !bundle.schedule_id) {
        throw new Error('server returned an incomplete schedule export');
      }
      const jsonBytes = new TextEncoder().encode(bundle.json_content);
      const pdfBytes = this._decodeSchedulePdf(bundle.pdf_base64);
      if (!Number.isSafeInteger(bundle.json_size_bytes)
          || jsonBytes.length !== bundle.json_size_bytes) {
        throw new Error('server JSON size does not match its export manifest');
      }
      if (!Number.isSafeInteger(bundle.pdf_size_bytes)
          || pdfBytes.length !== bundle.pdf_size_bytes) {
        throw new Error('server PDF size does not match its export manifest');
      }
      let scheduleDocument;
      try { scheduleDocument = JSON.parse(bundle.json_content); }
      catch (_e) { throw new Error('server returned invalid schedule JSON'); }
      if (scheduleDocument.schedule_id !== bundle.schedule_id
          || scheduleDocument.schema_version !== bundle.schema_version) {
        throw new Error('server schedule identity is inconsistent');
      }
      await Promise.all([
        this._verifyScheduleDigest(jsonBytes, bundle.json_sha256, 'JSON'),
        this._verifyScheduleDigest(pdfBytes, bundle.pdf_sha256, 'PDF')
      ]);
      if (token !== this._scheduleExportToken) return;
      this._downloadScheduleBlob(
        new Blob([jsonBytes], { type: 'application/json;charset=utf-8' }),
        bundle.json_filename,
        'acquisition-schedule.json'
      );
      this._downloadScheduleBlob(
        new Blob([pdfBytes], { type: 'application/pdf' }),
        bundle.pdf_filename,
        'acquisition-schedule.pdf'
      );
      const completeness = bundle.complete ? 'NO CAP TRUNCATION' : 'DWELL CAP REACHED';
      const shortId = String(bundle.schedule_id).slice(0, 12).toUpperCase();
      this.setState({
        scheduleExportBusy: false,
        scheduleExportStatus: `${completeness} · ${bundle.dwell_count} DWELLS · ID ${shortId}`,
        scheduleExportError: !bundle.complete
      });
    } catch (e) {
      if (token !== this._scheduleExportToken) return;
      const message = String(e && e.message ? e.message : e).replace(/\s+/g, ' ').slice(0, 120);
      this.setState({
        scheduleExportBusy: false,
        scheduleExportStatus: `EXPORT FAILED · ${message}`,
        scheduleExportError: true
      });
    } finally {
      if (token === this._scheduleExportToken) this._scheduleExportActive = false;
    }
  }

  reset() {
    clearInterval(this._timer); clearInterval(this._auto); this._auto = null;
    this.setState({ t: 0, playing: false, stepIndex: 0, acquired: false, acqT: null, revealed: false, ...this._clearScheduleExportPatch(), tickN: this.state.tickN + 1 }, () => { this.setup(); this.rebuildStations(); this.updateGlobe(true); this.forceUpdate(); });
  }

  // formatting
  hms(s) { s = Math.max(0, Math.round(s)); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), x = s % 60; return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(x).padStart(2,'0')}`; }
  mmss(s) { s = Math.max(0, Math.round(s)); return `${Math.floor(s / 60)}m${String(s % 60).padStart(2,'0')}s`; }
  segStyle(active, color) {
    const c = color || '#7aa6f0';
    return `padding:7px 11px;font-family:'IBM Plex Sans',sans-serif;font-size:10px;font-weight:500;letter-spacing:.04em;cursor:pointer;border-radius:5px;border:1px solid ${active ? c : 'rgba(255,255,255,.12)'};background:${active ? 'rgba(122,166,240,.14)' : 'transparent'};color:${active ? '#dce6fb' : '#7b828d'}`;
  }

  renderVals() {
    if (!this.P) this.setup();
    const P = this.P, st = this.state, t = st.t, disp = this.displayGrid(t);
    let mean = 0; for (let k = 0; k < P.nGrid; k++) mean += disp[k] * this.deltas[k];
    let varr = 0; for (let k = 0; k < P.nGrid; k++) varr += disp[k] * (this.deltas[k] - mean) ** 2;
    const sigma = Math.sqrt(varr), ent = this.entropy(disp), nonDet = this.log.filter(l => !l.detect).length;
    // The live session supplies entropy when the backend bridge is enabled.
    const live = (window.LEOPT_LIVE && this.live && this.live.entropy != null) ? this.live : null;
    const entShown = live ? live.entropy : ent;

    const ended = this.schedule && st.stepIndex >= this.schedule.length;
    const acqStr = st.acquired ? '+' + this.hms(st.acqT) : (ended ? 'NO ACQ' : 'SEARCHING');
    const acqColor = st.acquired ? '#6bbf94' : (ended ? '#d8786e' : '#e0a96b');
    const statCells = [
      { label: 'ACQUISITION', value: acqStr, color: acqColor },
      { label: 'DWELLS EXEC', value: String(this.log.length), color: '#dadee4' },
      { label: 'NON-DETECT', value: String(nonDet), color: nonDet > 0 ? '#d8959c' : '#dadee4' },
      { label: '1\u03c3 ALONG-TRK', value: sigma.toFixed(2) + '\u00b0', color: '#e0a96b' },
      { label: live ? 'ENTROPY ◉ LIVE' : 'ENTROPY', value: entShown.toFixed(2), color: live ? '#7aa6f0' : '#dadee4' },
      { label: 'MAX GAP', value: this.mmss(this.maxGap), color: this.maxGap > 1800 ? '#d8959c' : '#dadee4' }
    ];

    const slot = this.schedule ? this.schedule[st.stepIndex] : null;
    let cd = {}; const showDwell = !st.acquired && !!slot;
    if (slot) {
      const aim = this.planAim(slot, this.displayGrid(slot.t)), g = this.geomQuality(slot.pass);
      const spA = this.subpoint(this.uNow(slot.t) + aim, slot.t);
      const cc = this.centralAngle(slot.station.lat, slot.station.lng, spA.lat, spA.lng);
      const el = Math.max(0, this.elevation(cc)), az = this.azimuth(slot.station.lat, slot.station.lng, spA.lat, spA.lng);
      const pd = this.expectedDetect(aim, g, this.displayGrid(slot.t));
      cd = { cdStation: slot.station.name, cdTime: this.hms(slot.t), cdAz: az.toFixed(0), cdEl: el.toFixed(0), cdHold: slot.hold, cdAim: (aim >= 0 ? '+' : '') + aim.toFixed(2) + '\u00b0', cdPdet: (pd * 100).toFixed(0) + '%', cdPdetW: Math.round(pd * 100), cdPdetColor: this.pStroke(pd, 0.85) };
    }

    const rows = (this.stations || []).map(s => ({
      code: s.code,
      passes: (this.passesByStation[s.name] || []).map(p => {
        const active = t >= p.tAOS && t <= p.tLOS;
        return { leftPct: (p.tAOS / P.window * 100).toFixed(2), widthPct: Math.max(0.5, (p.tLOS - p.tAOS) / P.window * 100).toFixed(2), bg: p.__bg, border: active ? '#eef0f3' : p.__border, glow: active ? '0 0 0 1px rgba(238,240,243,.4)' : 'none' };
      })
    }));
    const axisTicks = []; for (let h = 0; h <= Math.round(P.window / 3600); h++) axisTicks.push({ label: h + 'h' });

    const hist = this.history || [{ t: 0, ent: 0 }];
    let emin = Infinity, emax = -Infinity; for (const h of hist) { emin = Math.min(emin, h.ent); emax = Math.max(emax, h.ent); }
    if (emax - emin < 0.1) emax = emin + 1;
    const sparkPath = hist.map((h, i) => { const x = hist.length <= 1 ? 0 : (i / (hist.length - 1)) * 200; const y = 34 - ((h.ent - emin) / (emax - emin)) * 30; return x.toFixed(1) + ',' + y.toFixed(1); }).join(' ');

    // planner fields
    const f = this.form || {};
    const fld = (key, label, unit) => ({ label, unit: unit || '', value: String(f[key] ?? ''), on: e => this.setField(key, e.target.value) });
    const orbitFields = [fld('satName', 'DESIGNATOR'), fld('alt', 'ALTITUDE', 'km'), fld('inc', 'INCLINATION', '\u00b0'), fld('raan', 'RAAN', '\u00b0'), fld('u0', 'ARG-LAT @ SEP', '\u00b0'), fld('sigma0', 'INIT 1\u03c3', '\u00b0')];
    const groundFields = [fld('beamBeta', 'BEAM WIDTH', '\u00b0'), fld('holdNom', 'DWELL HOLD', 's')];
    const simFields = [fld('windowH', 'LEOP WINDOW', 'h'), fld('growth', '\u03c3 GROWTH', '\u00b0/h'), fld('minEl', 'MIN ELEVATION', '\u00b0'), fld('truthDeg', 'HYPOTH. TRUTH', '\u00b0')];
    // Spacecraft TT&C antenna presets (mirror acquisition_platform.satellite_antennas);
    // drives the recommended LEOP dwell/hold.
    const SAT_ANTS = [
      { id: 'turnstile', label: 'TURNSTILE' }, { id: 'canted_turnstile', label: 'CANTED' },
      { id: 'dual_omni', label: 'DUAL-OMNI' }, { id: 'monopole', label: 'MONOPOLE' },
      { id: 'patch_single', label: 'PATCH' }
    ];
    const satSegments = SAT_ANTS.map(a => ({
      label: a.label, on: () => this.setField('satAnt', a.id), style: this.segStyle((f.satAnt || 'turnstile') === a.id)
    }));
    const advTab = st.advTab || 'orbit';
    const advTabs = [['orbit', 'ORBIT'], ['antenna', 'ANTENNA'], ['satellite', 'SATELLITE'], ['simulation', 'SIMULATION']]
      .map(([k, label]) => ({ label, on: () => this.setAdvTab(k), style: this.advTabStyle(advTab === k) }));
    const showOrbit = advTab === 'orbit', showAntenna = advTab === 'antenna',
      showSatellite = advTab === 'satellite', showSimulation = advTab === 'simulation';
    const _p = this.parseOrbit(f.opm || '');
    const _ok = _p.inc != null || _p.alt != null || _p.raan != null;
    const _bits = [
      _p.inc != null ? _p.inc.toFixed(1) + '\u00b0 INC' : null,
      _p.alt != null ? _p.alt + ' km' : null,
      _p.raan != null ? 'RAAN ' + _p.raan.toFixed(1) + '\u00b0' : null,
      (_p.ecc != null && isFinite(_p.ecc)) ? 'e ' + _p.ecc.toFixed(4) : null,
    ].filter(Boolean).join('   \u00b7   ');
    const parsedSummary = _ok
      ? `${_p.formatLabel || 'PARSED'} \u2713   ${_bits}`
      : (f.opm && f.opm.trim() ? 'Unrecognised \u2014 expected a TLE or CCSDS OPM (KVN).' : 'No orbit yet \u2014 paste a TLE/OPM, drop a file, or load the sample.');
    const stationRows = (f.stations || []).map((s, i) => ({
      name: String(s.name ?? ''), lat: String(s.lat ?? ''), lng: String(s.lng ?? ''),
      onName: e => this.setStationField(i, 'name', e.target.value), onLat: e => this.setStationField(i, 'lat', e.target.value), onLng: e => this.setStationField(i, 'lng', e.target.value), onRemove: () => this.removeStation(i)
    }));
    const planStrategies = [
      { label: 'AUTO', on: () => this.setField('strategy', 'auto'), style: this.segStyle(f.strategy === 'auto') },
      { label: 'INFO-GREEDY', on: () => this.setField('strategy', 'infogreedy'), style: this.segStyle(f.strategy === 'infogreedy') },
      { label: 'PFT', on: () => this.setField('strategy', 'pft'), style: this.segStyle(f.strategy === 'pft') },
      { label: 'SWEEP', on: () => this.setField('strategy', 'sweep'), style: this.segStyle(f.strategy === 'sweep') }
    ];
    // Ground-antenna agility presets (mirror planner.controller_select mounts);
    // the mount drives which controller "AUTO" selects.
    const MOUNTS = [
      { id: 'phased_array', label: 'PHASED ARRAY' },
      { id: 'agile_pedestal', label: 'AGILE' },
      { id: 'standard_dish', label: 'DISH' },
      { id: 'large_dish', label: 'LARGE' },
      { id: 'legacy_slow', label: 'SLOW' }
    ];
    const mountSegments = MOUNTS.map(m => ({
      label: m.label, on: () => this.setMount(m.id), style: this.segStyle((f.mount || 'standard_dish') === m.id)
    }));
    // The controller actually chosen by the backend (from the last live plan),
    // else the local preview from /api/recommend-controller.
    const ctl = (this.live && this.live.controller) || this._rec || null;
    const controllerNote = ctl && ctl.rationale ? ctl.rationale : '';
    const recNote = this._rec && this._rec.rationale
      ? (this._rec.is_agile ? '⚡ ' : '⚙ ') + this._rec.rationale
      : '';
    const presets = [
      { label: 'POLAR DEFAULT', on: () => this.applyPreset('default') },
      { label: 'SOUTHERN', on: () => this.applyPreset('south') },
      { label: 'MINIMAL', on: () => this.applyPreset('min') }
    ];
    const planSummary = `${(f.stations || []).length} STATIONS \u00b7 ${f.windowH || '?'}h WINDOW \u00b7 ${f.alt || '?'} km \u00b7 ${(f.strategy || '').toUpperCase()}`;
    const scheduleExportBusy = !!st.scheduleExportBusy;
    const scheduleExportStatus = st.scheduleExportStatus || '';
    const scheduleExportLabel = scheduleExportBusy
      ? 'SERVER EXPORT \u2026'
      : '\u2193 SERVER SCHEDULE \u00b7 JSON + PDF';
    const scheduleExportStatusColor = scheduleExportStatus.startsWith('DWELL CAP REACHED')
      ? '#e0a96b'
      : (st.scheduleExportError ? '#d8786e' : (scheduleExportStatus ? '#6bbf94' : '#646b76'));
    const scheduleExportStyle = `margin-left:0;padding:7px 12px;background:${scheduleExportBusy ? 'rgba(122,166,240,.08)' : 'transparent'};border:1px solid ${scheduleExportBusy ? 'rgba(122,166,240,.45)' : 'rgba(255,255,255,.12)'};border-radius:5px;color:${scheduleExportBusy ? '#9bbcf3' : '#aab0ba'};font-family:inherit;font-size:9px;letter-spacing:.08em;cursor:${scheduleExportBusy ? 'wait' : 'pointer'};opacity:${scheduleExportBusy ? '.72' : '1'}`;

    return {
      globeRef: el => { this.globeEl = el; },
      splashGlobeRef: el => { this.splashGlobeEl = el; },
      clockStr: this.hms(t), windowStr: this.hms(P.window), windowSec: Math.round(P.window), clockVal: t, clockPct: (t / P.window * 100).toFixed(2),
      acqStr, acqColor,
      acqDetail: st.acquired ? `${this.log.length} dwells \u00b7 ${nonDet} non-detections ruled out` : '',
      entStr: entShown.toFixed(2), minElStr: '\u2265' + Math.round(P.minEl) + '\u00b0 EL', maxGapStr: this.mmss(this.maxGap),
      statCells, rows, axisTicks, sparkPath,
      acquired: st.acquired, showDwell, ...cd,
      ...this.strategyBadge(st, ctl),
      revealLabel: st.revealed ? 'TRUTH \u25c9 SHOWN' : 'REVEAL TRUTH', revealStyle: this.segStyle(st.revealed, '#d8786e'),
      truthStr: (st.truthDeg >= 0 ? '+' : '') + st.truthDeg.toFixed(1) + '\u00b0', truthVal: st.truthDeg,
      playLabel: st.playing ? '\u2759\u2759 PAUSE' : '\u25b6 PLAY', playStyle: this.segStyle(st.playing) + ';padding:8px 14px',
      autoLabel: this._auto ? '\u25a0 STOP AUTO' : '\u21bb AUTO-RUN LOOP', autoStyle: this.segStyle(!!this._auto, '#e0a96b') + ';padding:8px 12px',
      planOpen: st.mode === 'plan', chromeOpen: st.mode === 'rehearse',
      orbitFields, groundFields, simFields, satSegments, advTabs, showOrbit, showAntenna, showSatellite, showSimulation, stationRows, planStrategies, mountSegments, controllerNote, recNote, presets, planSummary,
      opmValue: f.opm || '', parsedSummary, advOpen: st.advOpen, advCaret: st.advOpen ? '\u25be' : '\u25b8',
      onOpm: e => this.setOpm(e.target.value), onLoadSample: () => this.loadSampleOrbit(), onToggleAdv: () => this.toggleAdv(),
      fileInputRef: el => { this.fileInput = el; }, onBrowse: () => this.openFilePicker(), onPickFile: e => this.readPickedFile(e),
      showEdit: st.mode === 'rehearse', onEditPlan: () => this.setMode('plan'),
      scheduleExportBusy, scheduleExportStatus, scheduleExportLabel, scheduleExportStatusColor, scheduleExportStyle,
      onExportSchedule: () => this.exportServerSchedule(),
      splashOpen: st.splashVisible, splashOpacity: st.launching ? 0 : 1, splashScale: st.launching ? '1.06' : '1', splashPointer: st.launching ? 'none' : 'auto', onLaunch: () => this.launch(),
      enterOpacity: st.splashShown ? '1' : '0', enterShift: st.splashShown ? '0px' : '18px',
      onCancelPlan: () => this.setMode('rehearse'), onRollout: () => this.rollout(), onAddStation: () => this.addStation(),
      planBack: st.rolledOut, dragOver: st.dragOver, dropBorder: st.dragOver ? 'rgba(122,166,240,.6)' : 'rgba(255,255,255,.12)',
      onDragOver: e => { e.preventDefault(); if (!this.state.dragOver) this.setState({ dragOver: true }); },
      onDragLeave: e => { e.preventDefault(); this.setState({ dragOver: false }); },
      onDropFile: e => this.readDroppedFile(e),
      onPlay: () => this.togglePlay(), onAutoplay: () => this.autoplay(), onReset: () => this.reset(),
      onScrub: e => this.scrub(e), onSetTruth: e => this.setTruth(e), onToggleReveal: () => this.toggleReveal(),
      onDetect: () => this.report(true), onNoDetect: () => this.report(false)
    };
  }

  autoplay() {
    if (this._auto) { clearInterval(this._auto); this._auto = null; this.setState({ tickN: this.state.tickN + 1 }); return; }
    clearInterval(this._timer); this.setState({ playing: false, mode: 'rehearse' });
    this._auto = setInterval(() => {
      const slot = this.schedule[this.state.stepIndex];
      if (!slot || this.state.acquired) { clearInterval(this._auto); this._auto = null; this.setState({ tickN: this.state.tickN + 1 }); return; }
      const aim = this.planAim(slot, this.displayGrid(slot.t)), g = this.geomQuality(slot.pass);
      this.report(Math.random() < this.pdet(this.state.truthDeg, aim, g));
    }, 950);
  }
  report(detect) {
    const slot = this.schedule[this.state.stepIndex]; if (!slot) return; const tt = slot.t;
    this.grid = this.displayGrid(tt); this.tLast = tt;
    const aim = this.planAim(slot, this.grid), g = this.geomQuality(slot.pass);
    for (let k = 0; k < this.P.nGrid; k++) { const pd = this.pdet(this.deltas[k], aim, g); this.grid[k] *= detect ? pd : (1 - pd); }
    this.normalize(); this.log.push({ slot, aim, detect }); this.history.push({ t: tt, ent: this.entropy(this.grid) });
    let acquired = this.state.acquired, acqT = this.state.acqT;
    if (detect && !acquired) { acquired = true; acqT = tt; }
    this.computeGanttColors();
    this.setState({ stepIndex: this.state.stepIndex + 1, acquired, acqT, t: tt }, () => this.updateGlobe(true));
    this._reportLive(detect);
  }

  // Fold the same detect/no-detect into any live engine session, including one
  // created on demand for export. Failures remain non-fatal to the visual demo.
  async _reportLive(detect) {
    if (!window.LeoptLive || !this.liveSessionId) return;
    const d = (this._liveDwells || [])[this._liveCursor];
    if (!d) return;
    const generation = this._livePlanGeneration || 0;
    this.setState({ ...this._clearScheduleExportPatch(), tickN: this.state.tickN + 1 });
    this._liveCursor += 1;
    let task = null;
    try {
      task = window.LeoptLive.observe(
        this.liveSessionId, d.passIdx, d.dwellIdx, detect
      );
      this._liveMutationPromise = task;
      const live = await task;
      if (generation !== (this._livePlanGeneration || 0)) return;
      this.live = live;
      const cursor = this._liveCursor;
      this._liveDwells = [];
      for (const p of live.passes) for (const q of p.dwells) {
        this._liveDwells.push({ passIdx: p.idx, dwellIdx: q.idx, pDetect: q.pDetect });
      }
      this._liveCursor = Math.min(cursor, this._liveDwells.length);
      console.info('[leopt-live] observe', detect ? 'DETECT' : 'NO-DETECT',
        'pass', d.passIdx, 'dwell', d.dwellIdx, '| entropy', live.entropy != null ? live.entropy.toFixed(3) : '?');
      this.setState({ tickN: this.state.tickN + 1 });
    } catch (e) {
      console.warn('[leopt-live] observe failed:', e.message);
    } finally {
      if (task && this._liveMutationPromise === task) this._liveMutationPromise = null;
    }
  }

  render() {
    return window.renderLeoptTemplate(this.renderVals());
  }
}

window.LeoptConsole = LeoptConsole;
