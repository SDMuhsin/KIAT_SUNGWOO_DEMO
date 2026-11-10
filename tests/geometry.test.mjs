// node tests/geometry.test.mjs - pure geometry math (sampling + synthetic field)
import assert from 'node:assert/strict';
import * as M from '../js/scenes/geometry_math.js';
const S = M.generateSamples(20000, 7); let bad = 0;
for (let i = 0; i < 20000; i++) { const x = S.pos[i*3], y = S.pos[i*3+1], z = S.pos[i*3+2]; if (Math.abs(x) > 3.0001 || Math.abs(y) > 1.0001 || Math.abs(z) > 1.5001) bad++; if (S.kind[i] !== 4 && M.insideCell(x, z, -0.001) ) bad++; }
assert.equal(bad, 0, 'samples inside duct and outside cells');
for (const n of [1000, 4000, 20000]) { const c = M.countKinds(S.kind, n); assert.ok(Math.abs(c[0] / n - 0.7) < 0.05, 'prefix interior share ~70% at ' + n); assert.ok(c.every((v) => v > 0)); }
for (let i = 0; i < 100; i++) { const t = M.viridis(i / 99); assert.ok(t.every((v) => v >= 0 && v <= 1)); }
for (const ch of ['speed', 'pressure']) for (let i = 0; i < 200; i++) { const v = M.fieldValue01(-3 + 6 * (i % 20) / 19, 0.3, -1.4 + 2.8 * (i / 200), ch); assert.ok(v >= 0 && v <= 1 && Number.isFinite(v)); }
assert.ok(M.flowField(0, 0, 1.499).speed < M.flowField(-2.4, 0, 0).speed, 'no-slip-ish wall damping');
console.log('geometry tests pass');
