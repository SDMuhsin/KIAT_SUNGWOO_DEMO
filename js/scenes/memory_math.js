// memory_math.js - pure helpers for the memory scene (stages 5-7). No THREE, no DOM: runs in node.
// Everything is a stylised MODEL ESTIMATE built on js/model.js. Block sizes, op table, roofline classification and
// the step-time split all come from model.js numbers; the only constants added here are listed in ASSUMPTIONS.
import * as M from '../model.js';

/** Constants introduced by this scene (not in model.js). All stylised, all shown to the viewer as model estimate. */
export const ASSUMPTIONS = {
  launchUs: 3,                  // assumed kernel-launch / graph-node overhead per autograd node (microseconds)
  elemFwd: { flopPerElem: 10, bytesPerElemX: 2 },   // activation fn (tanh-like): 10 FLOP, read+write = 2 x dtype bytes
  elemBwd: { flopPerElem: 4, bytesPerElemX: 3 },    // derivative * incoming grad: 4 FLOP, 2 reads + 1 write
};

/** Volume per byte (world units^3) at zoom step 0, and the zoom quantum (volume ratio between steps). */
export const NU_REF = 2.2e-6;
/** Floor height of the memory blocks (world y); the roofline plot has its own floor (PLOT.floorY) far below so the two never overlap. */
export const MEM_FLOOR = -5.6;
export const STEP = 4;
export const nuOf = (n) => NU_REF / Math.pow(STEP, n);
export const LIMITS = { y: 2.4, z: 5.2, shelf: 8.6, gap: 1.1 };

/** Slab dims for one activation layer. Volume = nu * bytes exactly; thickness (x) ~ dtype bytes; y~features, z~batch. */
export function slabDims(cfg, layerBytes, dim, nu, dx) {
  const c = M.normalize(cfg), b = M.dtypeBytes(c.precision);
  const tBase = Math.min(0.95, Math.max(0.18, 0.62 * dx)), t = tBase * b / 4;
  const V = nu * layerBytes, A = V / t;
  const r = Math.min(5, Math.max(0.8, 0.55 * Math.pow(c.batch / Math.max(1, dim), 0.35)));
  const y = Math.sqrt(A / r), z = A / y;
  return { x: t, y, z, volume: V, bytes: layerBytes };
}
/** Cube edge of a parameter-type block (weights, gradients, Adam m, Adam v, extras). */
export const cubeEdge = (bytes, nu) => Math.cbrt(Math.max(0, nu * bytes));

/** Full block layout for a zoom level. extras = [bytes,...]. Returns slabs, edges and the constraints' maxima. */
export function layoutBlocks(cfg, extras, nu, dx) {
  const c = M.normalize(cfg), m = M.memoryBreakdown(c), dims = M.layerDims(c);
  const slabs = m.perLayerActivations.map((bytes, l) => slabDims(c, bytes, dims[l], nu, dx));
  const b = M.dtypeBytes(c.precision);
  const bytes = { weights: m.weights, gradients: m.gradients, adamM: m.params * 4, adamV: m.params * 4 };
  void b;
  const edges = { weights: cubeEdge(bytes.weights, nu), gradients: cubeEdge(bytes.gradients, nu), adamM: cubeEdge(bytes.adamM, nu), adamV: cubeEdge(bytes.adamV, nu), extras: extras.map((x) => cubeEdge(x, nu)) };
  const list = [edges.weights, edges.gradients, edges.adamM, edges.adamV, ...edges.extras];
  const shelf = list.reduce((a, e) => a + e, 0) + LIMITS.gap * (list.length - 1);
  const maxY = Math.max(...slabs.map((s) => s.y), ...list), maxZ = Math.max(...slabs.map((s) => s.z));
  return { slabs, edges, bytes, shelf, maxY, maxZ, memory: m };
}
/** Choose the integer zoom step n (nu = nuOf(n)) with hysteresis: zoom out when something overflows, in only with margin. */
export function pickScale(cfg, extras, dx, nPrev = 0) {
  const fits = (n, k) => { const l = layoutBlocks(cfg, extras, nuOf(n), dx); return l.maxY <= LIMITS.y * k && l.maxZ <= LIMITS.z * k && l.shelf <= LIMITS.shelf * k; };
  let n = nPrev; while (n < 14 && !fits(n, 1)) n++;
  while (n > -6 && fits(n - 1, 0.8)) n--;
  return n;
}

// ------------------------------------------------------------------ roofline / op table
export const OPS = [
  { id: 'mmFwd', name: 'matmul fwd', short: 'matmul fwd', lane: 0 },
  { id: 'mmBwd', name: 'matmul bwd', short: 'matmul bwd', lane: 1 },
  { id: 'deriv', name: 'derivative evaluation', short: 'derivatives', lane: 2 },
  { id: 'act', name: 'activation (elementwise)', short: 'activation', lane: 3 },
  { id: 'opt', name: 'optimizer update', short: 'optimizer', lane: 4 },
];

/** Plot geometry (world units). x = log10 arithmetic intensity; back wall y = log10 attained FLOP/s; lanes in z. */
export const PLOT = { xL: -1.2, xR: 9.0, floorY: -30.0, laneZ0: 1.9, laneDz: 0.72, bar: { z: 3.9 }, xO: 10.0, wallZ: -2.9, wallH: 3.7, aiLo: -1, aiHi: 3, perfLo: 10, perfHi: 14.8, hMax: 3.2 };
export const xOfAI = (ai) => PLOT.xL + (Math.min(PLOT.aiHi, Math.max(PLOT.aiLo, Math.log10(Math.max(1e-9, ai)))) - PLOT.aiLo) / (PLOT.aiHi - PLOT.aiLo) * (PLOT.xR - PLOT.xL);
export const yOfPerf = (flopsPerS) => PLOT.floorY + (Math.min(PLOT.perfHi, Math.max(PLOT.perfLo, Math.log10(Math.max(1, flopsPerS)))) - PLOT.perfLo) / (PLOT.perfHi - PLOT.perfLo) * PLOT.wallH;
export const laneZ = (lane) => PLOT.laneZ0 - lane * PLOT.laneDz;

/** Roofline polyline [{ai, perf}] n points from aiLo..aiHi decades: perf = min(peak, ai * bandwidth). */
export function roofCurve(cfg, n = 48) {
  const c = M.normalize(cfg), out = [];
  for (let i = 0; i < n; i++) { const ai = Math.pow(10, PLOT.aiLo + (PLOT.aiHi - PLOT.aiLo) * i / (n - 1)); out.push({ ai, perf: Math.min(c.peakTflops * 1e12, ai * c.bandwidthGBs * 1e9) }); }
  return out;
}

/**
 * Ops of one training step with roofline classification. variant: 'physics' (default, the PINN step) | 'supervised' | 'sage'.
 * Each op: {id, name, flops, bytes, intensity, tCompute, tMemory, time, bound:'memory'|'compute', perf}.
 * The elementwise activation kernels are modelled as an extra kernel pair on top of the model.js matmul phases (re-reading pre-activations). deriv = extra work of the derivative tower.
 * Overhead = (autograd nodes + 1 optimizer kernel) x ASSUMPTIONS.launchUs. Returns {ops, overhead, total, memShare, compShare, overShare, ridge, ...}.
 */
export function opTable(cfg, variant = 'physics') {
  const c = M.normalize(cfg), b = M.dtypeBytes(c.precision), A = ASSUMPTIONS;
  const sup = M.stepCost(c, 'supervised'), full = M.stepCost(c, variant), dims = M.layerDims(c);
  const nHid = c.batch * dims.slice(1, dims.length - 1).reduce((a, d) => a + d, 0);
  const elemBytesF = A.elemFwd.bytesPerElemX * b * nHid, elemBytesB = A.elemBwd.bytesPerElemX * b * nHid;
  const mk = (id, flops, bytes) => {
    const f = Math.max(0, flops), by = Math.max(1, bytes);
    const tC = f / (c.peakTflops * 1e12), tM = by / (c.bandwidthGBs * 1e9), time = Math.max(tC, tM);
    return { id, name: OPS.find((o) => o.id === id).name, flops: f, bytes: by, intensity: f / by, tCompute: tC, tMemory: tM, time, bound: tM >= tC ? 'memory' : 'compute', perf: time > 0 ? f / time : 0 };
  };
  const ops = [
    mk('mmFwd', sup.fwd.flops, sup.fwd.bytes),
    mk('mmBwd', sup.bwd.flops, sup.bwd.bytes),
    mk('deriv', variant === 'physics' ? full.fwd.flops + full.bwd.flops - sup.fwd.flops - sup.bwd.flops : 0,
      variant === 'physics' ? full.fwd.bytes + full.bwd.bytes - sup.fwd.bytes - sup.bwd.bytes : 1),
    mk('act', (A.elemFwd.flopPerElem + A.elemBwd.flopPerElem) * nHid, elemBytesF + elemBytesB),
    mk('opt', sup.opt.flops, sup.opt.bytes),
  ];
  const active = ops.filter((o) => o.flops > 0);
  const nodes = M.autogradGraph(c, variant).totalNodes + 1;
  const overhead = nodes * A.launchUs * 1e-6;
  const busy = active.reduce((a, o) => a + o.time, 0), total = busy + overhead;
  let tMem = 0, tComp = 0; for (const o of active) (o.bound === 'memory' ? (tMem += o.time) : (tComp += o.time));
  for (const o of ops) o.share = o.flops > 0 ? o.time / total : 0;
  return { variant, ops, overhead, overheadShare: overhead / total, launches: nodes, busy, total,
    memTime: tMem, compTime: tComp, memShare: tMem / total, compShare: tComp / total, overShare: overhead / total,
    memShareBusy: busy > 0 ? tMem / busy : 0, ridge: M.ridge(c), assumptions: A };
}

// ------------------------------------------------------------------ backward timeline
export const BWD = { wave: 0.6, optIn: [0.6, 0.76], upd: [0.78, 1.0] };
const ss = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
/**
 * Autograd backward timeline p in [0,1] over L layer boundaries (0 input .. L-1 output).
 * u = wavefront position in layer coordinates (L-1 -> 0). Activation l is saved until the wave reaches layer l (needed for
 * dW_{l+1} = a_l^T delta_{l+1} and sigma'(z_l)); then it is freed. Gradient of linear k appears as a_{k-1} is consumed.
 * Returns {u, wave (0..1 wave visible), keep[l], flash[l], gradFill (0..1 by param bytes), opt (0..1), upd (0..1)}.
 */
export function backwardState(p, L, paramsPerLinear) {
  const w = Math.min(1, Math.max(0, p / BWD.wave)), u = (L - 0.5) - (L + 0.3) * w;   // from just beyond the output layer to just beyond the input layer
  const keep = [], flash = [];
  for (let l = 0; l < L; l++) { keep.push(ss(l - 0.6, l + 0.3, u)); const d = (u - l) / 0.55; flash.push(w > 0 && w < 1 ? Math.exp(-d * d) : (w >= 1 && l === 0 ? Math.exp(-d * d) : 0)); }
  const tot = paramsPerLinear.reduce((a, x) => a + x, 0) || 1;
  let g = 0; for (let l = 0; l < L - 1; l++) g += paramsPerLinear[l] * (1 - keep[l]);
  const wave = p <= 0 ? 0 : w >= 1 ? Math.max(0, 1 - (p - BWD.wave) / 0.04) : 1;
  return { u, w, wave, keep, flash, gradFill: g / tot, opt: ss(BWD.optIn[0], BWD.optIn[1], p), upd: Math.min(1, Math.max(0, (p - BWD.upd[0]) / (BWD.upd[1] - BWD.upd[0]))) };
}
/** Parameter count of each linear layer k=1..L-1 (index k-1). */
export function paramsPerLinear(cfg) { const d = M.layerDims(cfg); return d.slice(0, -1).map((a, i) => a * d[i + 1] + d[i + 1]); }
