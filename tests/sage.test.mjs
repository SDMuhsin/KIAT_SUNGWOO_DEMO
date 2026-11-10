// node tests/sage.test.mjs - pure sage_math tests (tape, VJP code, timeline, collapse monotonicity, stage-14 series, results = SPEC facts)
import assert from 'node:assert/strict';
import * as M from '../js/model.js';
import * as S from '../js/scenes/sage_math.js';
let pass = 0; const t = (n, f) => { f(); pass++; console.log('ok  ', n); };

t('tape has exactly 40 entries, valid ops, every operand defined earlier', () => {
  const tp = S.buildTape(); assert.equal(tp.length, 40);
  tp.forEach((e) => { assert.ok(S.ruleIndex(e.op) >= 0, e.op); for (const a of [e.a, e.b]) if (a && /^t\d+$/.test(a)) assert.ok(+a.slice(1) < e.i, `${e.out} uses ${a}`); });
  assert.equal(S.RULES.length, 7);
});
t('reverse walk covers all 40 entries 39..0; generated code is a function with a return', () => {
  const w = S.reverseWalk(); assert.equal(w.length, 40); assert.equal(w[0].i, 39); assert.equal(w[39].i, 0);
  const code = S.genCode(); assert.ok(code[0].startsWith('def generated_backward')); assert.ok(code.at(-1).includes('return')); assert.ok(code.length > 90);
  assert.ok(code.some((l) => l.includes("g['DxT'] @")));
});
t('idea timeline phases are ordered and calls count up to the deck example', () => {
  const ph = [0, 4, 9, 15, 18, 25].map((x) => S.ideaState(x).phase); assert.deepEqual(ph, ['split', 'trace', 'reverse', 'emit', 'compile', 'loop']);
  assert.equal(S.ideaState(S.TL.trace[1]).tracedCount, 40); assert.equal(S.ideaState(S.TL_END + 10).calls, 30000); assert.equal(S.ideaState(10).calls, 0);
  let prev = -1; for (let x = 0; x < 30; x += 0.1) { const c = S.ideaState(x).walked; assert.ok(c >= prev); prev = c; }
});
t('overheadState: blend monotonically reduces AD memory, graph nodes, matmuls, step time; matmuls 24 -> 10', () => {
  for (const cfg of [{}, { depth: 8, pdeOrder: 3, nResiduals: 8 }, { depth: 1, pdeOrder: 1, nResiduals: 1, batch: 65536 }]) {
    let p = S.overheadState(cfg, 0);
    for (let b = 0.05; b <= 1.0001; b += 0.05) { const q = S.overheadState(cfg, b); assert.ok(q.memBytes <= p.memBytes + 1e-9 && q.adNodes <= p.adNodes && q.matmuls <= p.matmuls && q.total <= p.total + 1e-15 && q.totalNodes <= p.totalNodes, 'b=' + b); p = q; }
    const e = S.overheadState(cfg, 1); assert.equal(e.memBytes, 0); assert.equal(e.adNodes, 0); assert.ok(e.speedup > 1);
  }
  const d = S.overheadState({}, 0), a = S.overheadState({}, 1); assert.equal(d.matmuls, 24); assert.equal(a.matmuls, 10);
  assert.equal(d.adNodes, M.autogradGraph({}, 'physics').totalNodes); assert.ok(a.graphOverheadAfter < 0.2 * a.graphOverheadBefore);
});
t('jaxSeries: measured ONLY for the three deck figures; traffic 126 -> 31', () => {
  const a = S.jaxSeries({}), b = S.jaxSeries({ sageHalf: true }), c = S.jaxSeries({ sageHalf: true, sageSinglePass: true }), d = S.jaxSeries({ sageSinglePass: true });
  assert.deepEqual([a.ms, b.ms, c.ms], [12.7, 9.9, 7.7]); assert.ok(a.measured && b.measured && c.measured && !d.measured);
  assert.equal(a.units, 126); assert.ok(Math.abs(c.units - 31) < 1e-9); assert.ok(!b.trafficMeasured && a.trafficMeasured && c.trafficMeasured);
});
t('jaxState: toggles reduce model time; compile removes launch overhead; traffic moves the bandwidth-bound op', () => {
  const o = S.jaxState({}, {}), h = S.jaxState({}, { sageHalf: true }), hs = S.jaxState({}, { sageHalf: true, sageSinglePass: true }), cp = S.jaxState({}, { sageCompile: true });
  assert.ok(h.total < o.total && hs.total < h.total && cp.total < o.total && cp.overhead === 0 && o.overhead > 0);
  assert.equal(o.ops[0].bound, 'memory'); assert.ok(hs.ops[0].intensity > o.ops[0].intensity * 3.9);
  assert.ok(o.speedupVsAllOff === 1 && hs.speedupVsAllOff > 1);
});
t('results headline equals SPEC facts', () => {
  const h = S.headline(); assert.equal(h.perStep, '50x'); assert.equal(h.a40, '16x'); assert.equal(h.rtx, '~10x'); assert.equal(h.bedrock, '~40x'); assert.equal(h.pytorchMs, 303); assert.equal(h.oursMs, 6.1);
  const r = S.runRows(); assert.equal(r[0].before, 25.4); assert.equal(r[0].after, 1.6); assert.equal(r[1].labelBefore, '9–10 min'); assert.equal(r[2].before, 51); assert.equal(r[2].after, 1.3); assert.ok(r[2].projected);
});
t('amdahl: whole-run gain < per-step gain, decreasing in setup share, matches deck within rounding', () => {
  const k = M.results().perStep; let prev = 1e9;
  for (let s = 0; s <= 0.95; s += 0.05) { const a = S.amdahl(s, k); assert.ok(a <= k + 1e-9 && a < prev); prev = a; }
  assert.ok(Math.abs(S.amdahl(S.PRESET_SHARE.fine, k) - 16) < 3); assert.ok(Math.abs(S.amdahl(S.PRESET_SHARE.bedrock, k) - 40) < 4);
  const b = S.amdahlBars(0.66, k); assert.ok(Math.abs(b.pytorch.setup + b.pytorch.steps - 1) < 1e-9);
});
console.log(pass + ' sage tests passed');
