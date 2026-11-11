// Run: node tests/model.test.mjs   (pure model + layout + timeline; no DOM, no THREE)
import assert from 'node:assert/strict';
import * as M from '../demo/js/model.js';
import { makeBuffers, fillGraph, GR } from '../demo/js/graph.js';
import { timeline, prog } from '../demo/js/anim.js';
import { drawCount } from '../demo/js/cloud.js';

let n = 0; const t = (name, fn) => { fn(); n++; console.log('ok', name); };
const base = M.DEFAULTS;

t('params/memory monotone in width, depth, batch', () => {
  for (const k of ['width', 'depth', 'batch']) assert.ok(M.memoryBreakdown({ ...base, [k]: base[k] * 2 }).total > M.memoryBreakdown(base).total, k);
});
t('batch affects activations only; width affects weights; bf16 halves w/a/g', () => {
  assert.equal(M.memoryBreakdown({ ...base, batch: 8192 }).weights, M.memoryBreakdown(base).weights);
  const a = M.memoryBreakdown(base), b = M.memoryBreakdown({ ...base, precision: 'bf16' });
  assert.equal(b.weights * 2, a.weights); assert.equal(b.activations * 2, a.activations); assert.equal(b.optimizer, a.optimizer);
});
t('Adam off removes optimizer state and cost', () => {
  assert.equal(M.memoryBreakdown({ ...base, adam: false }).optimizer, 0);
  assert.ok(M.flops({ ...base, adam: false }).opt < M.flops(base).opt);
});
t('param count hand check; bwd = 2x fwd flops', () => {
  assert.equal(M.paramCount({ inDim: 3, outDim: 4, width: 8, depth: 1 }), 3 * 8 + 8 + 8 * 4 + 4);
  const f = M.flops(base); assert.equal(f.bwd, 2 * f.fwd);
});
t('SPEC: 24 vs 10 matmuls at default depth; 7 passes for 2D NS order 2', () => {
  assert.equal(M.autogradGraph(base, 'physics').matmuls, M.FACTS.matmuls.baseline);
  assert.equal(M.autogradGraph(base, 'sage').matmuls, M.FACTS.matmuls.sage);
  assert.equal(M.autogradGraph({ ...base, inDim: 2, pdeOrder: 2 }, 'physics').passes, M.FACTS.autogradPasses2dNS);
});
t('graph nodes: supervised is ~30 at default, physics is much larger, sage == supervised', () => {
  const s = M.autogradGraph(base, 'supervised'), p = M.autogradGraph(base, 'physics'), g = M.autogradGraph(base, 'sage');
  assert.equal(s.totalNodes, 30); assert.ok(p.totalNodes > 15 * s.totalNodes); assert.equal(g.totalNodes, s.totalNodes);
});
t('residual points: physics cost grows with them, SAGE does not blow up; default ratio is 1', () => {
  assert.equal(M.resRatio(base), 1);
  const lo = M.stepCost({ ...base, resPoints: 1024 }, 'physics'), hi = M.stepCost({ ...base, resPoints: 16384 }, 'physics');
  assert.ok(hi.total.time > lo.total.time); assert.ok(hi.memory.activationsStored > lo.memory.activationsStored);
});
t('sageEndpoints: SAGE strictly cheaper on every readout, across knob corners', () => {
  for (const c of [base, { ...base, depth: 1 }, { ...base, depth: 12, width: 1024 }, { ...base, pdeOrder: 3, resPoints: 262144 }, { ...base, pdeOrder: 1, resPoints: 64 }, { ...base, precision: 'bf16', adam: false }]) {
    const e = M.sageEndpoints(c);
    for (const k of ['memory', 'stepTime', 'nodes', 'matmuls']) assert.ok(e.sage[k] < e.base[k], k + ' ' + JSON.stringify(c));
  }
  const d = M.sageEndpoints(base); assert.equal(d.base.matmuls, 24); assert.equal(d.sage.matmuls, 10); assert.equal(d.sage.nodes, 30);
});
t('SAGE added costs: v = points x outputs x bytes, built-once is small and constant in batch/points, extra pass is cheap; net still cheaper', () => {
  const d = M.sageEndpoints(base), a = d.added;
  assert.equal(a.v, base.resPoints * base.outDim * 4); assert.equal(M.sageAdded({ ...base, precision: 'bf16' }).v, a.v / 2);
  assert.equal(a.tape, 40); assert.equal(M.sageAdded({ ...base, resPoints: 262144, batch: 262144 }).builtOnce, a.builtOnce);
  assert.ok(a.builtOnce < 1e5 && a.v + a.builtOnce < 0.05 * d.removed.graphBytes);
  assert.ok(a.extraTime > 0 && a.extraTime < d.sage.stepTimePlain);
  assert.equal(d.sage.memory, d.sage.memoryPlain + a.v + a.builtOnce); assert.equal(d.sage.stepTime, d.sage.stepTimePlain + a.extraTime);
  assert.ok(d.sage.memory < d.base.memory && d.sage.stepTime < d.base.stepTime); assert.equal(d.removed.nodes, 618);
  assert.ok(Math.abs(d.base.memory / 2 ** 20 - 154.9) < 0.5 && d.base.matmuls === 24 && d.sage.matmuls === 10);
});
t('equations knob: baseline graph/memory/time grow monotonically; SAGE grows only v and built-once; SAGE always cheaper', () => {
  const at = (n) => M.paramsFromKnobs((id) => (id === 'equations' ? n : undefined));
  assert.equal(M.towerFactor(at(4)), 19); assert.equal(M.autogradGraph(at(4), 'physics').totalNodes, 618);
  let prev = null;
  for (let n = 1; n <= 7; n++) {
    const c = at(n), e = M.sageEndpoints(c);
    assert.ok(e.sage.memory < e.base.memory && e.sage.stepTime < e.base.stepTime, 'eq ' + n);
    assert.equal(e.sage.nodes, 30); assert.equal(e.sage.matmuls, 10 + 0);
    if (prev) { for (const k of ['memory', 'stepTime', 'nodes']) assert.ok(e.base[k] > prev.base[k], k + n); assert.ok(e.added.builtOnce > prev.added.builtOnce); assert.ok(e.added.v >= prev.added.v); assert.ok(e.sage.memoryPlain >= prev.sage.memoryPlain); }
    prev = e;
  }
  assert.ok(at(7).outDim === 7 && at(3).outDim === 4);
  assert.equal(M.EQUATION_SETS.length, 7);
});
t('layout: instance counts equal the model node counts for all depth/order corners (no overflow)', () => {
  const buf = makeBuffers();
  for (const depth of [1, 4, 12]) for (const pdeOrder of [1, 2, 3]) for (const resPoints of [64, 4096, 262144]) for (const nResiduals of [1, 4, 7]) {
    const c = { ...base, depth, pdeOrder, resPoints, nResiduals };
    fillGraph(buf, c, 'physics', -9, -12); assert.equal(buf.nNodes, M.autogradGraph(c, 'physics').totalNodes);
    for (let i = 0; i < buf.nNodes * 4; i++) assert.ok(Number.isFinite(buf.nodes.A[i]));
    assert.ok(buf.nEdges > buf.nNodes / 2 && buf.nEdges < buf.edgeCap, 'edge capacity ' + buf.nEdges);
    const sb = makeBuffers(120, 400); fillGraph(sb, c, 'supervised', -5, -6); assert.equal(sb.nNodes, M.autogradGraph(c, 'supervised').totalNodes);
  }
});
t('node size follows memory: batch up -> bigger nodes; bf16 -> smaller', () => {
  const sz = (c) => { const b = makeBuffers(120, 400); fillGraph(b, c, 'supervised', 0, 0); let s = 0; for (let i = 0; i < b.nNodes; i++) s += b.nodes.A[i * 4]; return s; };
  assert.ok(sz({ ...base, batch: 65536 }) > sz(base)); assert.ok(sz({ ...base, precision: 'bf16' }) < sz(base));
});
t('timeline: windows ordered, physics backward longer than data backward, SAGE shortens the step', () => {
  const ends = M.sageEndpoints(base), t0 = timeline(ends, 0), t1 = timeline(ends, 1);
  for (const tl of [t0, t1]) { assert.ok(tl.pts[1] <= tl.fwd[0] + 1e-9 && tl.fwd[1] <= tl.out[0] + 1e-9 && tl.out[1] <= tl.bwdData[0] + 1e-9 && tl.netBwd[1] <= tl.opt[0] + 1e-9 && tl.opt[1] <= 1); }
  assert.ok((t0.bwdPhys[1] - t0.bwdPhys[0]) * t0.total > 1.5 * (t0.bwdData[1] - t0.bwdData[0]) * t0.total);
  assert.ok(t1.total < t0.total); assert.ok(!t0.hasVp && t1.hasVp && t1.vp[1] > t1.vp[0] && t1.pts[1] <= t1.vp[0] + 1e-9 && t1.vp[1] <= t1.fwd[0] + 1e-9); assert.equal(prog(0.5, [0.2, 0.4]), 1);
});
t('point counts follow knobs monotonically within the pool', () => {
  assert.ok(drawCount(64) < drawCount(4096) && drawCount(4096) < drawCount(262144) && drawCount(262144) <= 2560);
});
t('FACTS from the decks are intact', () => {
  assert.equal(M.FACTS.traffic.baseline, 126); assert.equal(M.FACTS.stepMs.sage, 7.7);
  const r = M.results(); assert.ok(Math.abs(r.perStep - 49.7) < 0.2);
});
t('paramsFromKnobs maps ids', () => {
  const c = M.paramsFromKnobs((id) => ({ bandwidth: 100, width: 64, adam: false, resPoints: 128 })[id]);
  assert.equal(c.bandwidthGBs, 100); assert.equal(c.width, 64); assert.equal(c.adam, false); assert.equal(c.resPoints, 128); assert.equal(c.depth, base.depth);
});
console.log(`\n${n} tests passed`);
