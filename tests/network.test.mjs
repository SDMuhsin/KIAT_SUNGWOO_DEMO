// node tests/network.test.mjs - pure network math
import assert from 'node:assert/strict';
import * as N from '../js/scenes/network_math.js';
import * as model from '../js/model.js';
const s = N.layerSizes(128, 4); assert.deepEqual(s.true, [3, 128, 128, 128, 128, 4]); assert.deepEqual(s.shown, [3, 16, 16, 16, 16, 4]);
assert.deepEqual(N.layerSizes(8, 2).shown, [3, 8, 8, 4]);
for (const act of Object.keys(N.ACTS)) {
  const w1 = N.makeWeights(s.shown, act), w2 = N.makeWeights(s.shown, act);
  assert.deepEqual(Array.from(w1.W[1]), Array.from(w2.W[1]), 'seeded determinism');
  const inp = new Float32Array([0.1, -0.5, 0.9, -1, 1, 0]);
  const r = N.forward(s.shown, w1, inp, 2);
  assert.equal(r.N, 3 + 64 + 4);
  assert.ok(r.acts.every(Number.isFinite));
  // independent recompute of point 0 output
  let a = Array.from(inp.slice(0, 3));
  for (let l = 0; l < s.shown.length - 1; l++) { const o = []; for (let j = 0; j < s.shown[l + 1]; j++) { let t = w1.b[l][j]; for (let i = 0; i < a.length; i++) t += w1.W[l][j * a.length + i] * a[i]; o.push(l === s.shown.length - 2 ? t : N.ACTS[act](t)); } a = o; }
  for (let j = 0; j < 4; j++) assert.ok(Math.abs(a[j] - r.acts[r.off[5] + j]) < 1e-4, 'forward matches reference');
  // hidden activations are not degenerate
  const h = Array.from(r.acts.slice(r.off[2], r.off[3])); assert.ok(h.some((v) => Math.abs(v) > 0.05) && h.some((v) => Math.abs(v) < 0.98 || act !== 'tanh'), 'non-degenerate ' + act);
}
let c = 0; for (let i = 0; i < 400; i++) c += N.sampleByMagnitude(new Float32Array([5, 0, 0, 0]), 0, 4, i / 400) === 0 ? 1 : 0; assert.ok(c > 300, 'magnitude sampling favours large |a|');
assert.ok(N.weightMag(100, 16, 1.4) === 1 && N.weightMag(0, 16, 1.4) === 0);
assert.equal(model.paramCount({ width: 128, depth: 4, inDim: 3, outDim: 4 }), 3 * 128 + 128 + 3 * (128 * 128 + 128) + 128 * 4 + 4);
const g = N.gridDims(16); assert.equal(g.rows * g.cols, 16);
console.log('network tests pass');
