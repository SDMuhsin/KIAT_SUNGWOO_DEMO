// network_math.js - pure helpers for the network scene (node-testable). Everything here is ILLUSTRATIVE:
// seeded synthetic weights of the DRAWN (capped) network, not trained weights.
export const CAP = 16;                 // max displayed neurons per hidden layer
export const ACTS = {
  tanh: Math.tanh,
  sin: Math.sin,
  silu: (x) => x / (1 + Math.exp(-x)),
};
export const ACT_GAIN = { tanh: 1.4, sin: 1.8, silu: 1.6 };

export function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function gauss(rnd) { let u = 0, v = 0; while (u === 0) u = rnd(); while (v === 0) v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

/** Layer sizes: true (model) and drawn (capped). Layers: input, depth hidden, output. */
export function layerSizes(width, depth, inDim = 3, outDim = 4, cap = CAP) {
  const trueS = [inDim, ...Array(depth).fill(width), outDim];
  const shown = trueS.map((n, i) => (i === 0 || i === trueS.length - 1 ? n : Math.min(n, cap)));
  return { true: trueS, shown };
}

/** Seeded weights for the drawn network. W[l] is (shown[l+1] x shown[l]) row-major, b[l] is shown[l+1]. */
export function makeWeights(shown, act = 'tanh', seed = 1234) {
  const gain = ACT_GAIN[act] ?? 1.4, W = [], b = [];
  for (let l = 0; l < shown.length - 1; l++) {
    const nin = shown[l], nout = shown[l + 1], rnd = mulberry32(seed + 7919 * (l + 1));
    const w = new Float32Array(nin * nout), bb = new Float32Array(nout);
    const s = (l === 0 ? 1.6 : gain) / Math.sqrt(nin);
    for (let i = 0; i < w.length; i++) w[i] = gauss(rnd) * s;
    for (let i = 0; i < nout; i++) bb[i] = gauss(rnd) * 0.15;
    W.push(w); b.push(bb);
  }
  return { W, b, act, gain };
}

export function offsets(shown) { const o = [0]; for (const n of shown) o.push(o[o.length - 1] + n); return o; }

/**
 * Real forward pass of the drawn network for P inputs (xyz in [-1,1]).
 * Returns acts: Float32Array(P * N) with N = sum(shown); per-point layout = layers concatenated. Output layer is linear.
 */
export function forward(shown, net, inputs, P, out) {
  const off = offsets(shown), N = off[off.length - 1], f = ACTS[net.act] ?? Math.tanh;
  const acts = out && out.length >= P * N ? out : new Float32Array(P * N);
  const L = shown.length;
  for (let p = 0; p < P; p++) {
    const base = p * N;
    for (let k = 0; k < shown[0]; k++) acts[base + k] = inputs[p * 3 + k];
    for (let l = 0; l < L - 1; l++) {
      const nin = shown[l], nout = shown[l + 1], w = net.W[l], bb = net.b[l], ia = base + off[l], oa = base + off[l + 1];
      for (let j = 0; j < nout; j++) {
        let s = bb[j]; const r = j * nin;
        for (let i = 0; i < nin; i++) s += w[r + i] * acts[ia + i];
        acts[oa + j] = l === L - 2 ? s : f(s);
      }
    }
  }
  return { acts, N, off };
}

/** Normalised |w| in 0..1 for edge thickness/opacity (relative to the layer's typical scale). */
export function weightMag(w, nin, gain) { return Math.min(1, Math.abs(w) * Math.sqrt(nin) / (gain * 2.6)); }

/** Sample a neuron index in [0,n) with probability proportional to |a|+eps, using u in [0,1). */
export function sampleByMagnitude(acts, base, n, u, eps = 0.05) {
  let tot = 0; for (let k = 0; k < n; k++) tot += Math.abs(acts[base + k]) + eps;
  let x = u * tot;
  for (let k = 0; k < n; k++) { x -= Math.abs(acts[base + k]) + eps; if (x <= 0) return k; }
  return n - 1;
}

/** Grid layout (rows x cols) for n neurons of a layer plane. */
export function gridDims(n) { if (n <= 4) return { rows: n, cols: 1 }; const cols = n >= 12 ? 4 : n >= 6 ? 3 : 2; return { rows: Math.ceil(n / cols), cols }; }
