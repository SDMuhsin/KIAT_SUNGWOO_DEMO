// node tests/memory.test.mjs - pure memory-scene math (memory_math.js on top of model.js)
import assert from 'node:assert/strict';
import * as MM from '../js/scenes/memory_math.js';
import * as M from '../js/model.js';

// block volume = bytes * nu exactly (slabs), cube edge^3 = nu*bytes
for (const cfg of [{}, { batch: 65536, width: 512, depth: 8 }, { precision: 'bf16' }]) {
  const nu = MM.nuOf(0), lay = MM.layoutBlocks(cfg, [], nu, 1.4), m = M.memoryBreakdown(cfg);
  lay.slabs.forEach((s, l) => { assert.ok(Math.abs(s.x * s.y * s.z - nu * m.perLayerActivations[l]) / (nu * m.perLayerActivations[l]) < 1e-9, 'slab volume ∝ bytes'); });
  assert.ok(Math.abs(lay.edges.weights ** 3 - nu * m.weights) < 1e-9 && Math.abs(lay.edges.adamM ** 3 + lay.edges.adamV ** 3 - nu * m.optimizer) < 1e-9);
}
// bf16 thins slabs, halves weights volume
{ const a = MM.layoutBlocks({}, [], MM.nuOf(0), 1.4), b = MM.layoutBlocks({ precision: 'bf16' }, [], MM.nuOf(0), 1.4);
  assert.ok(Math.abs(b.slabs[2].x * 2 - a.slabs[2].x) < 1e-9 && Math.abs(b.slabs[2].y - a.slabs[2].y) < 1e-9); assert.ok(Math.abs(b.edges.weights ** 3 * 2 - a.edges.weights ** 3) < 1e-12); }
// zoom picks a level where everything fits, with hysteresis
for (const cfg of [{}, { batch: 262144, width: 1024 }, { batch: 64, width: 1024, depth: 12 }, { batch: 262144, width: 16, depth: 12 }]) {
  const dx = 7.35 / (cfg.depth ?? 4 + 1), n = MM.pickScale(cfg, [], dx, 0), l = MM.layoutBlocks(cfg, [], MM.nuOf(n), dx);
  assert.ok(l.maxY <= MM.LIMITS.y && l.maxZ <= MM.LIMITS.z && l.shelf <= MM.LIMITS.shelf, 'fits ' + JSON.stringify(cfg));
  assert.equal(MM.pickScale(cfg, [], dx, n), n, 'stable (hysteresis)');
}
// op table: shares sum to 1; classification flips with bandwidth/peak; bf16 raises AI; overhead shrinks with batch
{ const t = MM.opTable({}); const sum = t.ops.reduce((a, o) => a + o.share, 0) + t.overShare; assert.ok(Math.abs(sum - 1) < 1e-9); assert.ok(Math.abs(t.memShare + t.compShare + t.overShare - 1) < 1e-9);
  const by = (cfg, id) => MM.opTable(cfg).ops.find((o) => o.id === id);
  assert.equal(by({}, 'deriv').bound, 'memory'); assert.equal(by({ bandwidthGBs: 3000 }, 'deriv').bound, 'compute'); assert.equal(by({ peakTflops: 4 }, 'deriv').bound, 'compute');
  assert.equal(by({}, 'opt').bound, 'memory'); assert.equal(by({ peakTflops: 200, bandwidthGBs: 100 }, 'opt').bound, 'memory');
  assert.ok(by({ precision: 'bf16' }, 'deriv').intensity > by({}, 'deriv').intensity * 1.9);
  assert.ok(MM.opTable({ batch: 65536 }).overShare < MM.opTable({ batch: 512 }).overShare);
  assert.ok(MM.opTable({ batch: 65536 }).memShare > 0.8, 'large batch: memory-bound dominates (Sep profiling story)');
  assert.equal(MM.opTable({}, 'sage').ops.find((o) => o.id === 'deriv').flops, 0); assert.ok(MM.opTable({}, 'sage').total < MM.opTable({}, 'physics').total);
  for (const o of t.ops) assert.equal(o.bound, o.tMemory >= o.tCompute ? 'memory' : 'compute'); }
// roofline mapping monotone, ridge consistent with model.ridge
{ assert.ok(MM.xOfAI(10) > MM.xOfAI(1)); const c = MM.roofCurve({}, 20); assert.ok(c.every((p, i) => i === 0 || p.perf >= c[i - 1].perf)); assert.ok(Math.abs(c.at(-1).perf - 37e12) < 1);
  const r = M.ridge({}); assert.ok(c.filter((p) => p.ai < r).every((p) => Math.abs(p.perf - p.ai * 696e9) < 1)); }
// backward timeline: activations consumed output->input, gradients fill monotone to 1, opt after wave
{ const L = 6, ppl = MM.paramsPerLinear({}); let last = -1;
  for (let p = 0; p <= 1.0001; p += 0.02) { const s = MM.backwardState(p, L, ppl); assert.ok(s.gradFill >= last - 1e-9); last = s.gradFill; }
  const s0 = MM.backwardState(0, L, ppl), s1 = MM.backwardState(1, L, ppl); assert.ok(s0.keep.every((k) => k > 0.99) && s0.gradFill === 0 && s0.opt === 0);
  assert.ok(s1.keep.every((k) => k < 0.01) && Math.abs(s1.gradFill - 1) < 1e-9 && s1.opt > 0.99 && s1.upd > 0.99);
  const mid = MM.backwardState(0.3, L, ppl); assert.ok(mid.keep[L - 1] < 0.01 && mid.keep[0] > 0.99, 'output activation freed first, input last'); }
console.log('memory tests pass');
