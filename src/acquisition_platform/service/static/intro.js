// Animated logo reveal, followed by the repeating signal rings.
(function () {
'use strict';
const h = React.createElement;
const LOGO_SRC = 'leopt-logo.png';

const STAGE_W = 1280;
const STAGE_H = 720;

const LOGO_W = 820;
const LOGO_SCALE = LOGO_W / 1200;
const LOGO_H = 800 * LOGO_SCALE;
const LOGO_X = (STAGE_W - LOGO_W) / 2;
const LOGO_Y = 92;

const TARGET = { x: LOGO_X + 996 * LOGO_SCALE, y: LOGO_Y + 243 * LOGO_SCALE };

const CX = 637, CY = 338, A = 322, B = 112;
const ROT = -13 * Math.PI / 180;
const COSR = Math.cos(ROT), SINR = Math.sin(ROT);
const THETA0 = -35 * Math.PI / 180;

const clamp = (v, mn, mx) => Math.max(mn, Math.min(mx, v));
const E = {
  easeOutCubic: (t) => (--t) * t * t + 1,
  easeInOutCubic: (t) => (t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1),
};

function ellipsePt(th) {
  const ex = A * Math.cos(th), ey = B * Math.sin(th);
  return { x: CX + ex * COSR - ey * SINR, y: CY + ex * SINR + ey * COSR };
}
function lerpPt(a, b, e) { return { x: a.x + (b.x - a.x) * e, y: a.y + (b.y - a.y) * e }; }

const T_LIFT = 1.0, T_ORBIT = 1.3, T_ORBIT_END = 4.3, T_DOCK = 4.8;

function ballState(t) {
  if (t < T_LIFT) return { ...ellipsePt(THETA0), vis: 0, docked: false };
  if (t < T_ORBIT) {
    const p = clamp((t - T_LIFT) / (T_ORBIT - T_LIFT), 0, 1);
    const e = E.easeOutCubic(p);
    return { ...lerpPt(TARGET, ellipsePt(THETA0), e), vis: p, docked: false };
  }
  if (t < T_ORBIT_END) {
    const p = clamp((t - T_ORBIT) / (T_ORBIT_END - T_ORBIT), 0, 1);
    const e = E.easeInOutCubic(p);
    const th = THETA0 - 2 * 2 * Math.PI * e;
    return { ...ellipsePt(th), vis: 1, docked: false };
  }
  if (t < T_DOCK) {
    const p = clamp((t - T_ORBIT_END) / (T_DOCK - T_ORBIT_END), 0, 1);
    const e = E.easeOutCubic(p);
    return { ...lerpPt(ellipsePt(THETA0), TARGET, e), vis: 1, docked: false };
  }
  return { ...TARGET, vis: 1, docked: true };
}

function Logo({ t }) {
  const p = clamp((t - 0.2) / 0.8, 0, 1);
  const e = E.easeOutCubic(p);
  const scale = 0.9 + 0.1 * e;
  return h('img', {
    src: LOGO_SRC,
    alt: 'LEOPT',
    style: {
      position: 'absolute', left: LOGO_X, top: LOGO_Y, width: LOGO_W, height: LOGO_H,
      opacity: e, transform: `scale(${scale})`, transformOrigin: '50% 60%',
      filter: 'drop-shadow(0 8px 28px rgba(255,150,20,0.28))', willChange: 'transform, opacity',
    },
  });
}

function Shine({ t }) {
  const p = clamp((t - 4.7) / 1.1, 0, 1);
  const opacity = p > 0 && p < 1 ? 1 : 0;
  const x = -260 + p * (LOGO_W + 380);
  const maskUrl = `url(${LOGO_SRC})`;
  return h('div', {
    style: {
      position: 'absolute', left: LOGO_X, top: LOGO_Y, width: LOGO_W, height: LOGO_H,
      overflow: 'hidden', opacity,
      WebkitMaskImage: maskUrl, maskImage: maskUrl,
      WebkitMaskSize: `${LOGO_W}px ${LOGO_H}px`, maskSize: `${LOGO_W}px ${LOGO_H}px`,
      WebkitMaskRepeat: 'no-repeat', maskRepeat: 'no-repeat', pointerEvents: 'none',
    },
  }, h('div', {
    style: {
      position: 'absolute', top: '-30%', left: x, width: 150, height: '160%',
      background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.9), transparent)',
      transform: 'rotate(16deg)', filter: 'blur(2px)',
    },
  }));
}

function Pings({ t }) {
  const launches = [4.9, 5.55, 6.2, 6.85];
  const life = 1.15, speed = 150, ringMax = 4;
  return h('svg', {
    width: STAGE_W, height: STAGE_H,
    style: { position: 'absolute', inset: 0, pointerEvents: 'none' },
  }, launches.map((launch, i) => {
    const age = t - launch;
    if (age < 0 || age > life) return null;
    return h('circle', {
      key: i, cx: TARGET.x, cy: TARGET.y, r: age * speed,
      fill: 'none', stroke: 'rgba(255,205,90,1)', strokeWidth: ringMax * (1 - age / life) + 0.6,
      style: { opacity: (1 - age / life) * 0.7 },
    });
  }));
}

function CometTrail({ t }) {
  if (t < T_ORBIT + 0.05 || t > T_ORBIT_END + 0.15) return null;
  const count = 16, dt = 0.045;
  const dots = [];
  for (let k = 1; k <= count; k++) {
    const s = ballState(t - k * dt);
    if (!s.vis) continue;
    const f = 1 - k / count;
    const r = 3 + f * 7;
    dots.push(h('div', {
      key: k,
      style: {
        position: 'absolute', left: s.x, top: s.y,
        width: r * 2, height: r * 2, marginLeft: -r, marginTop: -r, borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(255,225,150,1) 0%, rgba(255,160,30,0.6) 55%, rgba(255,140,20,0) 75%)',
        opacity: f * f * 0.6,
      },
    }));
  }
  return h('div', { style: { position: 'absolute', inset: 0, pointerEvents: 'none' } }, dots);
}

function SatelliteBall({ t }) {
  const b = ballState(t);
  if (b.vis <= 0) return null;
  const dockPulse = b.docked ? 1 + 0.04 * Math.sin(t * 6) : 1;
  const r = 13 * dockPulse;
  return h('div', {
    style: {
      position: 'absolute', left: b.x, top: b.y,
      width: r * 2, height: r * 2, marginLeft: -r, marginTop: -r, borderRadius: '50%',
      background: 'radial-gradient(circle at 35% 30%, #fff6d8 0%, #ffce4a 35%, #ff9b08 70%, #e87a00 100%)',
      boxShadow: '0 0 14px 4px rgba(255,180,40,0.85), 0 0 38px 10px rgba(255,150,20,0.45)',
      opacity: b.vis, willChange: 'transform, left, top',
    },
  });
}

function LeoptIntro() {
  const [t, setT] = React.useState(0);
  React.useEffect(() => {
    const intro = 7.0, loopStart = 5.85, loopEnd = 13.0, loopSpan = loopEnd - loopStart;
    let raf, start = null;
    const step = (ts) => {
      if (start == null) start = ts;
      const elapsed = (ts - start) / 1000;
      setT(elapsed < intro ? elapsed : loopStart + ((elapsed - intro) % loopSpan));
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, []);
  return h('div', { style: { position: 'absolute', inset: 0, overflow: 'visible' } },
    h(Logo, { t }), h(Shine, { t }), h(Pings, { t }), h(CometTrail, { t }), h(SatelliteBall, { t }));
}

window.LeoptIntro = LeoptIntro;
})();
