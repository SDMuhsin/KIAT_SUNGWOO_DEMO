// model.js - pure analytic cost model for an MLP PINN training step. No imports, no DOM: runs in node.
// EVERYTHING here is a stylised "model estimate" (label it so in the HUD) except FACTS, which are
// quoted from the Sep/Oct 2026 decks. Do not present model output as a measurement.

/** Numbers taken verbatim from the decks (SPEC "Facts"). Anything derived from these is also a fact-derived number. */
export const FACTS = {
  traffic: { baseline: 126, sage: 31, unit: 'traffic units (Sep profiling)' },
  stepMs: { baseline: 12.7, sage: 7.7 },
  matmuls: { baseline: 24, sage: 10, refLayers: 5 },
  autogradPasses2dNS: 7,
  perStepMs: { pytorchUofW: 303, ours: 6.1 }, // Oct; 303/6.1 ~ 50x
  fineTune: { a40: { beforeMin: 25.4, afterMin: 1.6 }, rtx5080: { beforeMin: [9, 10], afterMin: 1 } },
  bedrock: { beforeH: 51, afterH: 1.3, steps: 600000, projected: true },
};

/** Default knob values / model config. Units: bandwidthGBs in GB/s, peakTflops in TFLOP/s (A40-like, illustrative). */
export const DEFAULTS = {
  batch: 4096, width: 128, depth: 4, // depth = number of HIDDEN layers; linear layers L = depth + 1
  inDim: 3, outDim: 4, precision: 'fp32',
  bandwidthGBs: 696, peakTflops: 37,
  pdeOrder: 2, nResiduals: 4,
  resPoints: 4096, // physics (residual) collocation points per step; batch = data points per step
  adam: true, lambda: 1, // Adam on/off; physics loss weight (visual only: no effect on cost)
  sageHalf: false, sageSinglePass: false,
};

/** Merge partial cfg over DEFAULTS. All other functions call this, so you can pass partial objects. */
export function normalize(cfg = {}) { return { ...DEFAULTS, ...cfg }; }

/** Build model cfg from a knob getter, e.g. paramsFromKnobs(ctx.knobs.get). Knob ids -> cfg keys. */
export function paramsFromKnobs(get) {
  const g = (id, d) => { const v = get(id); return v === undefined || v === null ? d : v; };
  return normalize({
    batch: g('batch', DEFAULTS.batch), width: g('width', DEFAULTS.width), depth: g('depth', DEFAULTS.depth),
    precision: g('precision', DEFAULTS.precision), bandwidthGBs: g('bandwidth', DEFAULTS.bandwidthGBs),
    peakTflops: g('peakTflops', DEFAULTS.peakTflops), pdeOrder: g('pdeOrder', DEFAULTS.pdeOrder),
    nResiduals: g('nResiduals', DEFAULTS.nResiduals), resPoints: g('resPoints', DEFAULTS.resPoints),
    adam: g('adam', true) !== false, lambda: g('lambda', 1),
    sageHalf: !!g('sageHalf', false), sageSinglePass: !!g('sageSinglePass', false),
  });
}

/** Bytes per element of the compute dtype ('fp32' -> 4, 'bf16' -> 2). Adam state is always fp32 in this model. */
export function dtypeBytes(precision) { return precision === 'bf16' ? 2 : 4; }

/** Layer widths [in, w, w, ..., w, out]; length = depth + 2, linear layers L = depth + 1. */
export function layerDims(cfg) {
  const c = normalize(cfg);
  return [c.inDim, ...Array(c.depth).fill(c.width), c.outDim];
}
/** Number of linear layers. */
export function linearLayers(cfg) { return normalize(cfg).depth + 1; }

/** Trainable parameter count (weights + biases). */
export function paramCount(cfg) {
  const d = layerDims(cfg); let n = 0;
  for (let i = 0; i < d.length - 1; i++) n += d[i] * d[i + 1] + d[i + 1];
  return n;
}

/**
 * Persistent memory blocks in bytes (the 3D "memory blocks" of stages 5-7).
 * weights/gradients in compute dtype; Adam m,v fp32 (8 B/param); activations = one stored tensor per layer
 * boundary (incl. input) x batch. perLayerActivations[i] = bytes of boundary i (use for stacked block heights).
 * Returns {weights, activations, gradients, optimizer, total, params, perLayerActivations}.
 */
export function memoryBreakdown(cfg) {
  const c = normalize(cfg), b = dtypeBytes(c.precision), p = paramCount(c), dims = layerDims(c);
  const perLayerActivations = dims.map((d) => c.batch * d * b);
  const activations = perLayerActivations.reduce((a, x) => a + x, 0);
  const weights = p * b, gradients = p * b, optimizer = c.adam ? p * 8 : 0;
  return { weights, activations, gradients, optimizer, total: weights + activations + gradients + optimizer, params: p, perLayerActivations };
}

/** Forward matmul FLOPs = 2*B*sum(in*out); backward = 2x forward (dW and dX); optimizer ~10 FLOP/param (Adam). */
export function flops(cfg) {
  const c = normalize(cfg), d = layerDims(c); let mac = 0;
  for (let i = 0; i < d.length - 1; i++) mac += d[i] * d[i + 1];
  const fwd = 2 * c.batch * mac;
  return { fwd, bwd: 2 * fwd, opt: (c.adam ? 10 : 2) * paramCount(c), total: 3 * fwd + (c.adam ? 10 : 2) * paramCount(c) };
}

/**
 * Stylised derivative-tower multiplier T = 1 + d * (2^(p+1) - 2): each input dim (d = inDim) needs derivatives of
 * order 1..p, each order roughly doubling the graph. T(d=3,p=2) = 19. A model assumption, not a measurement.
 */
export function towerFactor(cfg) { const c = normalize(cfg); return 1 + c.inDim * (Math.pow(2, c.pdeOrder + 1) - 2); }
/** Residual-point ratio rho = resPoints / batch (1 if resPoints is unset). */
export function resRatio(cfg) { const c = normalize(cfg); return c.resPoints == null ? 1 : c.resPoints / c.batch; }
/** Effective tower multiplier: only the residual-point fraction pays the derivative tower. T_eff = 1 + (T-1) * rho. */
export function towerEff(cfg) { return 1 + (towerFactor(cfg) - 1) * resRatio(cfg); }

function phase(flopsN, bytesN, c) {
  const tC = flopsN / (c.peakTflops * 1e12), tM = bytesN / (c.bandwidthGBs * 1e9);
  return { flops: flopsN, bytes: bytesN, intensity: flopsN / bytesN, tCompute: tC, tMemory: tM, time: Math.max(tC, tM), bound: tM >= tC ? 'memory' : 'compute' };
}
/** Ridge point (FLOP/byte) where roofline switches from memory- to compute-bound. */
export function ridge(cfg) { const c = normalize(cfg); return (c.peakTflops * 1e12) / (c.bandwidthGBs * 1e9); }

/**
 * Roofline cost of one training step. variant: 'supervised' | 'physics' | 'sage'.
 *  supervised: plain MLP fwd+bwd+Adam.  physics: stored activations and fwd/bwd FLOPs x towerFactor (+ residual assembly).
 *  sage: one clean backprop (towerFactor = 1) + tiny closed-form residual vector.
 * Traffic model: fwd = W + 2A (write+read acts); bwd = W + 3A + G; opt = reads w,g,m,v writes w,m,v.
 * Returns {fwd, bwd, opt, total} each {flops, bytes, intensity, tCompute, tMemory, time(s), bound}, plus ridge, memory, variant.
 */
export function stepCost(cfg, variant = 'supervised') {
  const c = normalize(cfg), m = memoryBreakdown(c), f = flops(c), b = dtypeBytes(c.precision);
  const T = variant === 'physics' ? towerEff(c) : 1, rp = c.resPoints == null ? c.batch : c.resPoints;
  const resF = variant === 'supervised' ? 0 : rp * c.nResiduals * 8;
  const resB = variant === 'supervised' ? 0 : rp * c.nResiduals * b * (variant === 'physics' ? 4 : 2);
  const A = m.activations * T;
  const fwd = phase(f.fwd * T + resF, m.weights + 2 * A + resB, c);
  const bwd = phase(f.bwd * T, m.weights + 3 * A + m.gradients, c);
  const p = m.params;
  const opt = c.adam ? phase(f.opt, p * (b + b + 4 + 4) + p * (b + 4 + 4), c) : phase(f.opt, p * (b + b) + p * b, c);
  const flopsT = fwd.flops + bwd.flops + opt.flops, bytesT = fwd.bytes + bwd.bytes + opt.bytes;
  const total = phase(flopsT, bytesT, c);
  total.time = fwd.time + bwd.time + opt.time; // phases run sequentially
  total.bound = (fwd.tMemory + bwd.tMemory + opt.tMemory) >= (fwd.tCompute + bwd.tCompute + opt.tCompute) ? 'memory' : 'compute';
  return { variant, fwd, bwd, opt, total, ridge: ridge(c), memory: { ...m, activationsStored: A } };
}

/**
 * Autograd graph size (stylised node counts) + matmul count.
 * Supervised: 3 nodes/layer fwd (matmul,bias,act; last layer 2) + loss, backward mirrors it.
 * Physics: forwardNodes = F*towerFactor + 6*nResiduals; backward mirrors (double-backward). passes = 1 + d*p + 2
 *   (calibrated so inDim=2,p=2 gives the 7 passes of the Feb 2D-NS example).
 * SAGE: supervised-size graph only; PDE part is generated straight-line code, 0 graph nodes.
 * matmuls: supervised 3L-1; physics 4L+4 (=24 at L=5, SPEC); sage 2L (=10 at L=5, SPEC). Physics matmul count is
 * only depth-scaled; node count is the pdeOrder/nResiduals-sensitive quantity.
 * storedTensors ~ intermediates alive for backward. Returns {forwardNodes, backwardNodes, totalNodes, derivativeNodes, passes, matmuls, storedTensors, pdeGraphNodes}.
 */
export function autogradGraph(cfg, variant = 'physics') {
  const c = normalize(cfg), L = c.depth + 1, F = 3 * L; // 3(L-1)+2 net nodes + 1 loss
  if (variant === 'supervised' || variant === 'sage') {
    return { variant, forwardNodes: F, backwardNodes: F, totalNodes: 2 * F, derivativeNodes: 0, pdeGraphNodes: 0, passes: 2,
      matmuls: variant === 'sage' ? 2 * L : 3 * L - 1, storedTensors: L + 1 };
  }
  const T = towerFactor(c), fwdN = Math.round(F * T + 6 * c.nResiduals);
  return { variant, forwardNodes: fwdN, backwardNodes: fwdN, totalNodes: 2 * fwdN, derivativeNodes: fwdN - F,
    pdeGraphNodes: fwdN - F, passes: 1 + c.inDim * c.pdeOrder + 2, matmuls: 4 * L + 4, storedTensors: Math.round((L + 1) * T) };
}

/**
 * Endpoints of the SAGE toggle for the HUD (all MODEL ESTIMATES): {base, sage} each {memory (bytes), stepTime (s), nodes, matmuls}.
 * base = data graph + physics graph; sage = one ordinary graph (the physics graph is gone, its derivative part is the closed-form v).
 */
export function sageEndpoints(cfg) {
  const sup = autogradGraph(cfg, 'supervised'), phy = autogradGraph(cfg, 'physics'), sg = autogradGraph(cfg, 'sage');
  const pc = stepCost(cfg, 'physics'), sc = stepCost(cfg, 'sage');
  return {
    base: { memory: pc.memory.total - pc.memory.activations + pc.memory.activationsStored, stepTime: pc.total.time, nodes: sup.totalNodes + phy.totalNodes, matmuls: phy.matmuls, bwdTime: pc.bwd.time },
    sage: { memory: sc.memory.total, stepTime: sc.total.time, nodes: sg.totalNodes, matmuls: sg.matmuls, bwdTime: sc.bwd.time },
  };
}

/** Physics-vs-supervised inflation ratios {time, memory, nodes}, all model estimates. */
export function inflation(cfg) {
  const s = stepCost(cfg, 'supervised'), p = stepCost(cfg, 'physics');
  const gs = autogradGraph(cfg, 'supervised'), gp = autogradGraph(cfg, 'physics');
  return { time: p.total.time / s.total.time, memory: p.memory.total - p.memory.activations + p.memory.activationsStored, nodes: gp.totalNodes / gs.totalNodes,
    memoryRatio: (p.memory.total - p.memory.activations + p.memory.activationsStored) / s.memory.total };
}

/**
 * SAGE stage-14 numbers. Endpoints are SPEC facts: both off = 126 units / 12.7 ms, both on = 31 / 7.7.
 * The split between the two toggles is an ILLUSTRATIVE even geometric split (each toggle takes sqrt of the total
 * ratio); the decks do not give per-toggle numbers. Returns {trafficUnits, stepMs, illustrativeSplit:true}.
 */
export function sageTraffic({ sageHalf = false, sageSinglePass = false } = {}) {
  const n = (sageHalf ? 1 : 0) + (sageSinglePass ? 1 : 0), t = n / 2;
  return { trafficUnits: FACTS.traffic.baseline * Math.pow(FACTS.traffic.sage / FACTS.traffic.baseline, t),
    stepMs: FACTS.stepMs.baseline * Math.pow(FACTS.stepMs.sage / FACTS.stepMs.baseline, t), illustrativeSplit: n === 1 };
}

/** Oct results from FACTS: per-step speedup (~50x), fine-tune speedups (16x A40, ~10x RTX5080), bedrock ~40x PROJECTED. */
export function results() {
  const F = FACTS;
  return { perStep: F.perStepMs.pytorchUofW / F.perStepMs.ours,
    fineTuneA40: F.fineTune.a40.beforeMin / F.fineTune.a40.afterMin,
    fineTune5080: (F.fineTune.rtx5080.beforeMin[0] + F.fineTune.rtx5080.beforeMin[1]) / 2 / F.fineTune.rtx5080.afterMin,
    bedrockProjected: F.bedrock.beforeH / F.bedrock.afterH };
}

/** Human formatting helpers for HUD readouts. */
export function fmtBytes(n) { const u = ['B', 'KB', 'MB', 'GB', 'TB']; let i = 0; while (n >= 1024 && i < 4) { n /= 1024; i++; } return n.toFixed(n < 10 && i ? 2 : n < 100 && i ? 1 : 0) + ' ' + u[i]; }
export function fmtFlops(n) { const u = ['FLOP', 'KFLOP', 'MFLOP', 'GFLOP', 'TFLOP']; let i = 0; while (n >= 1000 && i < 4) { n /= 1000; i++; } return n.toFixed(n < 10 && i ? 2 : n < 100 && i ? 1 : 0) + ' ' + u[i]; }
export function fmtTime(s) { return s < 1e-3 ? (s * 1e6).toFixed(0) + ' us' : s < 1 ? (s * 1e3).toFixed(2) + ' ms' : s.toFixed(2) + ' s'; }
