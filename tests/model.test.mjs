// Run: node tests/model.test.mjs
import assert from 'node:assert/strict';
import * as M from '../js/model.js';

let n = 0; const t = (name, fn) => { fn(); n++; console.log('ok', name); };
const base = M.DEFAULTS;

t('params/memory monotone in width, depth, batch', () => {
  for (const k of ['width', 'depth', 'batch']) {
    const lo = M.memoryBreakdown({ ...base, [k]: base[k] }), hi = M.memoryBreakdown({ ...base, [k]: base[k] * 2 });
    assert.ok(hi.total > lo.total, k);
  }
  assert.ok(M.flops({ ...base, batch: 8192 }).total > M.flops(base).total);
});
t('batch affects activations only; width affects weights', () => {
  assert.equal(M.memoryBreakdown({ ...base, batch: 8192 }).weights, M.memoryBreakdown(base).weights);
  assert.ok(M.memoryBreakdown({ ...base, width: 256 }).weights > M.memoryBreakdown(base).weights);
});
t('bf16 halves weights/acts/grads, optimizer unchanged', () => {
  const a = M.memoryBreakdown(base), b = M.memoryBreakdown({ ...base, precision: 'bf16' });
  assert.equal(b.weights * 2, a.weights); assert.equal(b.activations * 2, a.activations);
  assert.equal(b.gradients * 2, a.gradients); assert.equal(b.optimizer, a.optimizer);
});
t('bwd flops = 2x fwd', () => { const f = M.flops(base); assert.equal(f.bwd, 2 * f.fwd); });
t('param count hand check', () => {
  assert.equal(M.paramCount({ inDim: 3, outDim: 4, width: 8, depth: 1 }), 3 * 8 + 8 + 8 * 4 + 4);
});
t('arithmetic intensity grows with width', () => {
  assert.ok(M.stepCost({ ...base, width: 512 }).fwd.intensity > M.stepCost({ ...base, width: 64 }).fwd.intensity);
});
t('bound flips: bandwidth up -> memory to compute', () => {
  const lo = M.stepCost({ ...base, bandwidthGBs: 50 }), hi = M.stepCost({ ...base, bandwidthGBs: 50000 });
  assert.equal(lo.fwd.bound, 'memory'); assert.equal(hi.fwd.bound, 'compute');
});
t('bound flips: peak flops down -> compute bound', () => {
  assert.equal(M.stepCost({ ...base, peakTflops: 0.05 }).fwd.bound, 'compute');
  assert.equal(M.stepCost({ ...base, peakTflops: 1000 }).fwd.bound, 'memory');
});
t('bound flips with width at default hardware', () => {
  assert.equal(M.stepCost({ ...base, width: 32 }).fwd.bound, 'memory');
  assert.equal(M.stepCost({ ...base, width: 1024 }).fwd.bound, 'compute');
});
t('time: more bandwidth never slower; bf16 not slower when memory-bound', () => {
  assert.ok(M.stepCost({ ...base, bandwidthGBs: 1400 }).total.time <= M.stepCost(base).total.time);
  assert.ok(M.stepCost({ ...base, precision: 'bf16' }).total.time < M.stepCost(base).total.time);
});
t('physics costs more than supervised, grows with pdeOrder & residuals', () => {
  const s = M.stepCost(base, 'supervised'), p = M.stepCost(base, 'physics');
  assert.ok(p.total.time > s.total.time); assert.ok(p.memory.activationsStored > s.memory.activations);
  assert.ok(M.stepCost({ ...base, pdeOrder: 3 }, 'physics').total.time > p.total.time);
  assert.ok(M.autogradGraph({ ...base, nResiduals: 8 }).totalNodes > M.autogradGraph(base).totalNodes);
});
t('graph nodes: physics > supervised, sage == supervised size, order monotone', () => {
  const s = M.autogradGraph(base, 'supervised'), p = M.autogradGraph(base, 'physics'), g = M.autogradGraph(base, 'sage');
  assert.ok(p.totalNodes > 5 * s.totalNodes); assert.equal(g.totalNodes, s.totalNodes); assert.equal(g.pdeGraphNodes, 0);
  assert.ok(M.autogradGraph({ ...base, pdeOrder: 1 }).totalNodes < p.totalNodes);
  assert.ok(M.autogradGraph({ ...base, depth: 8 }).totalNodes > p.totalNodes);
});
t('SPEC: 24 vs 10 matmuls at L=5; 7 passes for 2D NS order 2', () => {
  assert.equal(M.autogradGraph({ ...base, depth: 4 }, 'physics').matmuls, M.FACTS.matmuls.baseline);
  assert.equal(M.autogradGraph({ ...base, depth: 4 }, 'sage').matmuls, M.FACTS.matmuls.sage);
  assert.equal(M.autogradGraph({ ...base, inDim: 2, pdeOrder: 2 }, 'physics').passes, M.FACTS.autogradPasses2dNS);
});
t('SPEC: traffic 126->31, 12.7->7.7 ms, monotone in toggles', () => {
  const off = M.sageTraffic(), on = M.sageTraffic({ sageHalf: true, sageSinglePass: true }), one = M.sageTraffic({ sageHalf: true });
  assert.equal(off.trafficUnits, 126); assert.equal(off.stepMs, 12.7);
  assert.ok(Math.abs(on.trafficUnits - 31) < 1e-9); assert.ok(Math.abs(on.stepMs - 7.7) < 1e-9);
  assert.ok(one.trafficUnits < 126 && one.trafficUnits > 31); assert.ok(one.illustrativeSplit);
});
t('results: ~50x per step, 16x A40, bedrock ~40x', () => {
  const r = M.results();
  assert.ok(Math.abs(r.perStep - 49.7) < 0.2); assert.ok(Math.abs(r.fineTuneA40 - 15.9) < 0.1); assert.ok(Math.abs(r.bedrockProjected - 39.2) < 0.2);
});
t('paramsFromKnobs maps ids', () => {
  const c = M.paramsFromKnobs((id) => ({ bandwidth: 100, width: 64 })[id]);
  assert.equal(c.bandwidthGBs, 100); assert.equal(c.width, 64); assert.equal(c.depth, base.depth);
});
console.log(`\n${n} tests passed`);
