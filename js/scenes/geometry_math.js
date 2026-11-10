// geometry_math.js - pure (no THREE) helpers for the geometry scene: duct dims, seeded point sampling,
// a synthetic analytic flow-around-cylinders field, colormap, violation hotspots. Node-testable.
// ALL field values are ILLUSTRATIVE (stylised potential flow + wall damping), never a simulation result.

export const DUCT = { L: 6, H: 2, W: 3 };            // x (flow), y (up), z (span); box is centred on the origin
export const R = 0.42;                                // cell radius
export const CELLS = [{ x: -1.6, z: 0.5 }, { x: 0, z: -0.5 }, { x: 1.6, z: 0.5 }]; // cylinder axes along y
export const KINDS = ['interior', 'wall', 'inlet', 'outlet', 'obstacle'];
export const KIND_HEX = [0x56b4e9, 0xf0e442, 0x00b894, 0xcc79a7, 0xff6a1a]; // Okabe-Ito-ish, distinct from field colormap
export const MAX_POINTS = 20000;

export function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const hw = DUCT.L / 2, hh = DUCT.H / 2, hz = DUCT.W / 2;
export const insideCell = (x, z, pad = 0) => CELLS.some((c) => (x - c.x) ** 2 + (z - c.z) ** 2 < (R + pad) ** 2);

/**
 * generateSamples(max, seed): deterministic, order-independent prefix property (any prefix of n points has the same mix),
 * so a count knob is just a drawRange. Arrays are in duct-local coords; `tgt` is the normalised cube (-1..1)^3.
 * kinds: 0 interior, 1 wall, 2 inlet, 3 outlet, 4 obstacle surface. group: 0 interior, 1 boundary.
 */
export function generateSamples(max = MAX_POINTS, seed = 7) {
  const rnd = mulberry32(seed);
  const pos = new Float32Array(max * 3), tgt = new Float32Array(max * 3), kind = new Uint8Array(max), rand = new Float32Array(max), jit = new Float32Array(max * 3);
  for (let i = 0; i < max; i++) {
    let x, y, z, k;
    if (rnd() < 0.3) { // boundary
      const u = rnd();
      if (u < 0.5) { // walls: top/bottom (area 18 each) and sides (12 each), skip cell footprints
        k = 1;
        do {
          const a = rnd() * 60;
          if (a < 18) { y = hh; x = (rnd() * 2 - 1) * hw; z = (rnd() * 2 - 1) * hz; }
          else if (a < 36) { y = -hh; x = (rnd() * 2 - 1) * hw; z = (rnd() * 2 - 1) * hz; }
          else if (a < 48) { z = hz; x = (rnd() * 2 - 1) * hw; y = (rnd() * 2 - 1) * hh; }
          else { z = -hz; x = (rnd() * 2 - 1) * hw; y = (rnd() * 2 - 1) * hh; }
        } while ((y === hh || y === -hh) && insideCell(x, z, 0.02));
      } else if (u < 0.62) { k = 2; x = -hw; y = (rnd() * 2 - 1) * hh; z = (rnd() * 2 - 1) * hz; }
      else if (u < 0.74) { k = 3; x = hw; y = (rnd() * 2 - 1) * hh; z = (rnd() * 2 - 1) * hz; }
      else { k = 4; const c = CELLS[Math.floor(rnd() * CELLS.length)], th = rnd() * Math.PI * 2; x = c.x + R * Math.cos(th); z = c.z + R * Math.sin(th); y = (rnd() * 2 - 1) * hh; }
    } else { // interior collocation: 70% uniform, 30% refined around the cells (where gradients are sharp)
      k = 0;
      if (rnd() < 0.3) {
        for (;;) {
          const c = CELLS[Math.floor(rnd() * CELLS.length)], th = rnd() * Math.PI * 2, r = R + 0.03 - Math.log(1 - rnd() * 0.98) * 0.16;
          x = c.x + r * Math.cos(th); z = c.z + r * Math.sin(th); y = (rnd() * 2 - 1) * hh * 0.98;
          if (Math.abs(x) < hw && Math.abs(z) < hz && !insideCell(x, z, 0.01)) break;
        }
      } else {
        do { x = (rnd() * 2 - 1) * hw * 0.99; y = (rnd() * 2 - 1) * hh * 0.98; z = (rnd() * 2 - 1) * hz * 0.98; } while (insideCell(x, z, 0.02));
      }
    }
    pos.set([x, y, z], i * 3); tgt.set([x / hw, y / hh, z / hz], i * 3); kind[i] = k; rand[i] = rnd();
    jit.set([rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1], i * 3);
  }
  return { pos, tgt, kind, rand, jit, max };
}

export function countKinds(kind, n) { const c = [0, 0, 0, 0, 0]; for (let i = 0; i < n; i++) c[kind[i]]++; return c; }

/** Illustrative velocity magnitude and pressure at duct-local (x,y,z): potential flow around the 3 cells + no-slip-ish wall damping. */
export function flowField(x, y, z) {
  let ux = 1, uz = 0;
  for (const c of CELLS) {
    const dx = x - c.x, dz = z - c.z, r2 = Math.max(dx * dx + dz * dz, R * R), r4 = r2 * r2;
    ux += R * R * (dz * dz - dx * dx) / r4; uz += -2 * R * R * dx * dz / r4;
  }
  const d = Math.min(hz - Math.abs(z), hh - Math.abs(y)), s = Math.min(1, Math.max(0, d / 0.4)), wall = 0.22 + 0.78 * (1 - (1 - s) * (1 - s));
  const sp = Math.hypot(ux, uz) * wall;
  const p = 1 - 0.55 * (x + hw) / DUCT.L - 0.5 * sp * sp * 0.6;
  return { speed: sp, pressure: p };
}
/** Normalisation ranges (computed once on a coarse grid) so both channels map to 0..1. */
let _rng = null;
export function fieldRange() {
  if (_rng) return _rng;
  let s0 = 1e9, s1 = -1e9, p0 = 1e9, p1 = -1e9;
  for (let i = 0; i < 40; i++) for (let k = 0; k < 20; k++) {
    const x = -hw + (i + 0.5) / 40 * DUCT.L, z = -hz + (k + 0.5) / 20 * DUCT.W; if (insideCell(x, z)) continue;
    const f = flowField(x, 0, z); s0 = Math.min(s0, f.speed); s1 = Math.max(s1, f.speed); p0 = Math.min(p0, f.pressure); p1 = Math.max(p1, f.pressure);
  }
  return (_rng = { s0, s1, p0, p1 });
}
export function fieldValue01(x, y, z, channel = 'speed') {
  const f = flowField(x, y, z), r = fieldRange();
  const v = channel === 'pressure' ? (f.pressure - r.p0) / (r.p1 - r.p0) : (f.speed - r.s0) / (0.8 * (r.s1 - r.s0));
  return Math.pow(Math.min(1, Math.max(0, v)), 0.75);
}

const STOPS = [[0.267, 0.005, 0.329], [0.255, 0.267, 0.529], [0.165, 0.471, 0.557], [0.133, 0.659, 0.518], [0.992, 0.906, 0.145]];
/** viridis-like colormap, t in 0..1 -> [r,g,b] 0..1 (display space). Writes into out if given. */
export function viridis(t, out = [0, 0, 0]) {
  t = Math.min(1, Math.max(0, t)) * 4; const i = Math.min(3, Math.floor(t)), f = t - i, a = STOPS[i], b = STOPS[i + 1];
  out[0] = a[0] + (b[0] - a[0]) * f; out[1] = a[1] + (b[1] - a[1]) * f; out[2] = a[2] + (b[2] - a[2]) * f; return out;
}

/** Violation hotspots (duct-local): where an untrained/data-only net would break the PDE (cell flanks, wakes, near walls). */
export function hotspots() {
  const rnd = mulberry32(99), out = [];
  for (const c of CELLS) {
    out.push({ x: c.x, y: (rnd() - 0.5) * 1.2, z: c.z + R + 0.12, s: 0.7 + 0.3 * rnd() });
    out.push({ x: c.x, y: (rnd() - 0.5) * 1.2, z: c.z - R - 0.12, s: 0.7 + 0.3 * rnd() });
    out.push({ x: c.x + R + 0.55, y: (rnd() - 0.5) * 1.0, z: c.z, s: 0.8 + 0.2 * rnd() });
  }
  out.push({ x: -2.6, y: 0.5, z: 1.3, s: 0.5 }, { x: 2.5, y: -0.4, z: -1.3, s: 0.55 }, { x: 0.9, y: 0.3, z: 1.35, s: 0.5 });
  return out;
}
/** Smooth hotspot intensity at (x,z) in 0..1 (for the red slice overlay). */
export function hotspotIntensity(spots, x, z) {
  let v = 0; for (const h of spots) { const d2 = (x - h.x) ** 2 + (z - h.z) ** 2; v += h.s * Math.exp(-d2 / (2 * 0.3 * 0.3)); }
  return Math.min(1, v);
}
