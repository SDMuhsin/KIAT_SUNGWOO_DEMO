// physics_math.js - pure helpers for the physics scene (stages 8-11). No THREE, no DOM: runs in node.
// Everything numeric is a stylised MODEL ESTIMATE built on js/model.js (node counts equal model.autogradGraph exactly).
import * as M from '../model.js';
import { ASSUMPTIONS } from './memory_math.js';

export const EQUATIONS = [ // RANS k-epsilon, concise. `ord` = highest derivative order of u the term needs.
  { id: 'mass', name: 'Continuity', tex: '\u2207\u00b7u = 0', ord: 1 },
  { id: 'momx', name: 'Momentum x', tex: '(u\u00b7\u2207)u\u1d62 + \u2202\u1d62p/\u03c1 \u2212 \u2207\u00b7[(\u03bd+\u03bd\u209c)\u2207u\u1d62] = 0', ord: 2 },
  { id: 'momy', name: 'Momentum y', tex: 'same, i = y', ord: 2 },
  { id: 'momz', name: 'Momentum z', tex: 'same, i = z', ord: 2 },
  { id: 'k', name: 'Turbulent k', tex: 'u\u00b7\u2207k \u2212 \u2207\u00b7[(\u03bd+\u03bd\u209c/\u03c3k)\u2207k] \u2212 P + \u03b5 = 0', ord: 2 },
  { id: 'eps', name: 'Dissipation \u03b5', tex: 'u\u00b7\u2207\u03b5 \u2212 \u2207\u00b7[(\u03bd+\u03bd\u209c/\u03c3\u03b5)\u2207\u03b5] \u2212 (\u03b5/k)(C1P \u2212 C2\u03b5) = 0', ord: 2 },
  { id: 'nut', name: 'Eddy viscosity', tex: '\u03bd\u209c \u2212 C\u03bc k\u00b2/\u03b5 = 0', ord: 0 },
  { id: 'extra', name: 'Extra term', tex: 'e.g. energy / species balance', ord: 2 },
];

/** Branch (derivative-unit) counts per order: [1, 2d, 4d, ...]; their sum is model.towerFactor(cfg) exactly. */
export function branchCounts(cfg) {
  const c = M.normalize(cfg), out = [1];
  for (let k = 1; k <= c.pdeOrder; k++) out.push(c.inDim * Math.pow(2, k));
  return out;
}

/** Lane stream counts + names for the stage-9 tower. */
export const ORDER_NAMES = ['u', 'uₓ uᵧ u_z', 'uₓₓ uₓᵧ …', 'uₓₓₓ …'];

/** Training progress p (0..1) and lambda -> normalised losses. Data always fits; physics only decays as fast as lambda lets it. */
export function lossCurves(p, lambda) {
  const kPhys = 4.5 * lambda / (lambda + 0.4);
  const data = 0.06 + 0.94 * Math.exp(-5 * p);
  const phys = 0.08 + 0.92 * Math.exp(-kPhys * p);
  return { data, phys, violation: phys };
}

/** Stacked-bar heights (world units) at progress p; normalised by the initial total so the top starts at `top`. */
export function barHeights(p, lambda, nRes, top = 2.2) {
  const c = lossCurves(p, lambda), wn = nRes / 4;
  const total0 = 1 + lambda * wn, k = top / total0;
  return { data: c.data * k, phys: lambda * wn * c.phys * k, perTerm: lambda * c.phys * k * 0.25, total0: total0 * k };
}

// ------------------------------------------------------------------------------------------------------------------
// Autograd graph builder (stage 10 / 11 / SAGE reuse). Node ids are a valid topological order of the FORWARD graph.
// Slots per layer: 0 = matmul, 1 = bias, 2 = activation (last layer slot 2 = loss).
// ------------------------------------------------------------------------------------------------------------------
export const SLOT_COUNT = 3;

/** Closed-form counts for a given cfg/variant (used to size GPU buffers). */
export function graphCounts(cfg, variant = 'physics') {
  const c = M.normalize(cfg), L = c.depth + 1, bc = variant === 'physics' ? branchCounts(c) : [1], T = bc.reduce((a, x) => a + x, 0);
  const R = variant === 'physics' ? 6 * c.nResiduals : 0, F = SLOT_COUNT * L;
  const chain = (F - 1) * T, cross = variant === 'physics' ? 2 * L * (T - 1) : 0, res = R ? 3 * R + (R - 1) : 0;
  const fwdN = F * T + R, fwdE = chain + cross + res;
  const touch = (variant === 'physics' ? 2 * L * T + L * (T - 1) : 2 * L);
  return { L, T, R, fwdN, nodes: 2 * fwdN, fwdE, edges: 2 * fwdE + touch, touch };
}

/**
 * buildGraph(cfg, variant, {width}) -> arrays. width = x extent of one graph (world units) so graphs compare at the same scale.
 * nodes: x,y,z, ord (derivative order 0..3, 4 = residual assembly), bwd (0/1), slot, layer, t (backward arrival, in hops, for both twins)
 * edges: a,b (node ids), type (0 chain, 1 cross-branch, 2 residual merge, 3 saved-tensor touch), t (arrival of the later-lit endpoint), bwd
 */
export function buildGraph(cfg, variant = 'physics', { width = 8, yStep = 0.7, zStep = 0.4 } = {}) {
  const c = M.normalize(cfg), phys = variant === 'physics';
  const bc = phys ? branchCounts(c) : [1], T = bc.reduce((a, x) => a + x, 0), L = c.depth + 1, F = SLOT_COUNT * L, R = phys ? 6 * c.nResiduals : 0;
  const fwdN = F * T + R, N = 2 * fwdN;
  const sx = width / (SLOT_COUNT * L + 3), maxOrd = bc.length - 1;
  const H = (maxOrd + 1) * yStep + 1.0;                   // vertical offset forward -> backward copy
  const x = new Float32Array(N), y = new Float32Array(N), z = new Float32Array(N), ord = new Uint8Array(N), bwd = new Uint8Array(N), slot = new Uint8Array(N), layer = new Uint8Array(N), t = new Float32Array(N);
  const bOrd = [], bIdx = []; // per branch: order, index within order
  bc.forEach((cnt, k) => { for (let j = 0; j < cnt; j++) { bOrd.push(k); bIdx.push(j); } });
  const parentOf = (b) => { const k = bOrd[b]; if (k === 0) return -1; const off = bc.slice(0, k - 1).reduce((a, q) => a + q, 0); return off + (bIdx[b] % bc[k - 1]); };
  const x0 = -(SLOT_COUNT * L + 1) * sx / 2;
  const id = (b, l, s) => (b * L + l) * SLOT_COUNT + s;
  for (let b = 0; b < T; b++) {
    const k = bOrd[b], zz = (bIdx[b] - (bc[k] - 1) / 2) * zStep;
    for (let l = 0; l < L; l++) for (let s = 0; s < SLOT_COUNT; s++) {
      const i = id(b, l, s);
      x[i] = x0 + (l * SLOT_COUNT + s) * sx; y[i] = k * yStep; z[i] = zz; ord[i] = k; slot[i] = s; layer[i] = l;
    }
  }
  const xr = x0 + (SLOT_COUNT * L + 1.2) * sx;
  for (let j = 0; j < R; j++) { const i = F * T + j; x[i] = xr + (j % 2) * sx * 0.8; y[i] = (maxOrd * yStep) / 2 + ((j >> 1) % 3 - 1) * yStep * 0.9; z[i] = (Math.floor(j / 6) - (c.nResiduals - 1) / 2) * zStep * 2 + ((j >> 1) % 3 - 1) * 0.02; ord[i] = 4; slot[i] = 2; layer[i] = L - 1; }
  for (let i = 0; i < fwdN; i++) { const j = i + fwdN; x[j] = x[i]; y[j] = y[i] - H; z[j] = z[i]; ord[j] = ord[i]; bwd[j] = 1; slot[j] = slot[i]; layer[j] = layer[i]; }
  // forward edges
  const fa = [], fb = [], ft = [];
  const add = (a, b, ty) => { fa.push(a); fb.push(b); ft.push(ty); };
  for (let b = 0; b < T; b++) {
    for (let l = 0; l < L; l++) for (let s = 0; s < SLOT_COUNT; s++) { const i = id(b, l, s); if (s < 2) add(i, i + 1, 0); else if (l < L - 1) add(i, id(b, l + 1, 0), 0); }
    const p = parentOf(b);
    if (p >= 0) for (let l = 0; l < L; l++) { add(id(p, l, 1), id(b, l, 2), 1); add(id(p, l, 0), id(b, l, 1), 1); }
  }
  for (let j = 0; j < R; j++) {
    const i = F * T + j;
    for (const q of [(j * 7) % T, (j * 7 + 3) % T, (j * 7 + 11) % T]) add(id(q, L - 1, 2), i, 2);
    if (j > 0) add(i - 1, i, 2);
  }
  // arrival times (hops) of the backward sweep: longest path from the loss, over reversed edges
  const outs = Array.from({ length: fwdN }, () => []);
  fa.forEach((a, e) => outs[a].push(fb[e]));
  const tb = new Float32Array(fwdN);
  for (let i = fwdN - 1; i >= 0; i--) { let m = 0; for (const j of outs[i]) m = Math.max(m, tb[j] + 1); tb[i] = m; }
  let hops = 0; for (let i = 0; i < fwdN; i++) { t[i] = tb[i]; t[i + fwdN] = tb[i]; hops = Math.max(hops, tb[i]); }
  // edge list
  const E = fa.length, ea = [], eb = [], et = [], eT = [], eBwd = [];
  for (let e = 0; e < E; e++) { ea.push(fa[e]); eb.push(fb[e]); et.push(ft[e]); eT.push(tb[fb[e]] + 0.0); eBwd.push(0); }          // forward edge lit with the twin of its head
  for (let e = 0; e < E; e++) { ea.push(fb[e] + fwdN); eb.push(fa[e] + fwdN); et.push(ft[e]); eT.push(tb[fa[e]]); eBwd.push(1); } // backward edge: from head-twin to tail-twin
  const touch = (a, b) => { ea.push(a + fwdN); eb.push(b); et.push(3); eT.push(tb[a]); eBwd.push(1); };
  for (let b = 0; b < T; b++) for (let l = 0; l < L; l++) {
    touch(id(b, l, 0), id(b, l, 0)); touch(id(b, l, 2), id(b, l, 2));
    if (phys && parentOf(b) >= 0) touch(id(b, l, 0), id(parentOf(b), l, 2));
  }
  const edges = { a: Uint32Array.from(ea), b: Uint32Array.from(eb), type: Uint8Array.from(et), t: Float32Array.from(eT), bwd: Uint8Array.from(eBwd), count: ea.length };
  return { variant, L, T, fwdN, nodeCount: N, edgeCount: edges.count, hops, nodes: { x, y, z, ord, bwd, slot, layer, t }, edges,
    bounds: { w: width, h: H + (maxOrd + 1) * yStep, d: Math.max(1, (Math.max(...bc) - 1) * zStep, R ? c.nResiduals * zStep * 2 : 0), H }, sx, H };
}

/** Per-step time slabs (seconds, model estimate) for supervised vs physics: fwd/bwd/opt from model.stepCost plus autograd graph overhead. */
export function stepSlabs(cfg, variant) {
  const c = M.normalize(cfg), s = M.stepCost(c, variant), g = M.autogradGraph(c, variant), us = ASSUMPTIONS.launchUs * 1e-6;
  const build = g.forwardNodes * us, traverse = g.backwardNodes * us * (variant === 'physics' ? 1 : 1);
  const parts = { fwd: s.fwd.time, bwd: s.bwd.time, opt: s.opt.time, build, traverse };
  const total = parts.fwd + parts.bwd + parts.opt + build + traverse;
  return { variant, ...parts, total, graph: g, stepCost: s };
}
/** Physics-vs-supervised ratios {time, graphMemory(bytes), nodes}. */
export function costRatios(cfg) {
  const a = stepSlabs(cfg, 'supervised'), b = stepSlabs(cfg, 'physics');
  const extra = Math.max(0, b.stepCost.memory.activationsStored - b.stepCost.memory.activations);
  return { time: b.total / a.total, graphBytes: extra, memory: (b.stepCost.memory.total - b.stepCost.memory.activations + b.stepCost.memory.activationsStored) / a.stepCost.memory.total, nodes: b.graph.totalNodes / a.graph.totalNodes, sup: a, phys: b };
}
