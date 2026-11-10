// physics.test.mjs - pure physics_math tests (no deps): node tests/physics.test.mjs
import * as M from '../js/model.js';
import * as P from '../js/scenes/physics_math.js';
let pass = 0, fail = 0;
const t = (name, fn) => { try { fn(); pass++; console.log('ok  ', name); } catch (e) { fail++; console.log('FAIL', name, '-', e.message); } };
const eq = (a, b, m) => { if (a !== b) throw new Error(`${m ?? ''} ${a} !== ${b}`); };
const ok = (c, m) => { if (!c) throw new Error(m); };
const cfgs = [{}, { depth: 1, pdeOrder: 1, nResiduals: 1 }, { depth: 12, pdeOrder: 3, nResiduals: 8 }, { depth: 6, pdeOrder: 2, nResiduals: 2 }, { depth: 3, pdeOrder: 3, nResiduals: 5 }];

t('branchCounts sum equals model.towerFactor', () => { for (const c of cfgs) eq(P.branchCounts(c).reduce((a, x) => a + x, 0), M.towerFactor(c), JSON.stringify(c)); });
t('graph node counts equal model.autogradGraph (both variants)', () => {
  for (const c of cfgs) for (const v of ['supervised', 'physics']) { const g = P.buildGraph(c, v), m = M.autogradGraph(c, v); eq(g.nodeCount, m.totalNodes, v + JSON.stringify(c)); eq(g.fwdN, m.forwardNodes, 'fwd'); }
});
t('graphCounts matches built edge/node counts', () => {
  for (const c of cfgs) for (const v of ['supervised', 'physics']) { const g = P.buildGraph(c, v), k = P.graphCounts(c, v); eq(g.edgeCount, k.edges, 'edges ' + v + JSON.stringify(c)); eq(g.nodeCount, k.nodes, 'nodes'); }
});
t('forward ids are a topological order (every forward edge goes to a larger id)', () => {
  const g = P.buildGraph({}, 'physics'); for (let e = 0; e < g.edgeCount; e++) if (!g.edges.bwd[e] && g.edges.type[e] !== 3) ok(g.edges.a[e] < g.edges.b[e], 'edge ' + e);
});
t('physics graph is bigger and grows with order, terms, depth', () => {
  const base = P.buildGraph({ depth: 4, pdeOrder: 1, nResiduals: 2 }, 'physics').nodeCount, sup = P.buildGraph({ depth: 4 }, 'supervised').nodeCount;
  ok(base > sup, 'physics > supervised');
  ok(P.buildGraph({ depth: 4, pdeOrder: 2, nResiduals: 2 }, 'physics').nodeCount > base, 'order');
  ok(P.buildGraph({ depth: 4, pdeOrder: 1, nResiduals: 5 }, 'physics').nodeCount > base, 'terms');
  ok(P.buildGraph({ depth: 8, pdeOrder: 1, nResiduals: 2 }, 'physics').nodeCount > base, 'depth');
  ok(P.buildGraph({}, 'physics').nodeCount > 10 * P.buildGraph({}, 'supervised').nodeCount, 'at least 10x at defaults');
});
t('backward sweep takes more hops on the physics graph (same speed => slower)', () => {
  const a = P.buildGraph({}, 'supervised'), b = P.buildGraph({}, 'physics'); ok(b.hops > 2 * a.hops, `${b.hops} vs ${a.hops}`);
  ok(P.buildGraph({ pdeOrder: 3, nResiduals: 8 }, 'physics').hops > b.hops, 'grows with order/terms');
});
t('graph arrays are finite and twin nodes share arrival time', () => {
  const g = P.buildGraph({ depth: 5, pdeOrder: 3, nResiduals: 3 }, 'physics'), n = g.fwdN;
  for (let i = 0; i < g.nodeCount; i++) ok(Number.isFinite(g.nodes.x[i] + g.nodes.y[i] + g.nodes.z[i] + g.nodes.t[i]), 'finite ' + i);
  for (let i = 0; i < n; i++) eq(g.nodes.t[i], g.nodes.t[i + n], 'twin');
});
t('supervised and physics graphs have the same width (same scale)', () => {
  const w = (g) => { let a = 1e9, b = -1e9; for (let i = 0; i < g.nodeCount; i++) { if (g.nodes.ord[i] === 4) continue; a = Math.min(a, g.nodes.x[i]); b = Math.max(b, g.nodes.x[i]); } return b - a; };
  ok(Math.abs(w(P.buildGraph({}, 'supervised')) - w(P.buildGraph({}, 'physics'))) < 1e-4, 'width');
});
t('lossCurves: data decays, physics decays only with lambda > 0', () => {
  ok(P.lossCurves(1, 1).phys < 0.2 && P.lossCurves(0, 1).phys > 0.99, 'decay'); ok(P.lossCurves(1, 0).phys > 0.99, 'lambda 0 = no physics decay');
  ok(P.lossCurves(1, 5).phys <= P.lossCurves(1, 1).phys, 'monotone in lambda'); ok(P.lossCurves(1, 1).data < P.lossCurves(0, 1).data, 'data decays');
});
t('barHeights: top normalised at p=0, physics share grows with lambda', () => {
  const b = P.barHeights(0, 2, 4); ok(Math.abs(b.data + b.phys - 2.2) < 1e-9, 'top'); ok(P.barHeights(0, 5, 4).phys / 2.2 > P.barHeights(0, 0.5, 4).phys / 2.2, 'share'); eq(P.barHeights(0, 0, 4).phys, 0, 'lambda 0');
});
t('stepSlabs: physics step is slower and carries graph overhead', () => {
  const r = P.costRatios({}); ok(r.time > 3, 'ratio ' + r.time); ok(r.phys.build > r.sup.build * 5, 'overhead'); ok(r.graphBytes > 0, 'graph bytes');
  ok(P.costRatios({ pdeOrder: 3 }).time > P.costRatios({ pdeOrder: 1 }).time, 'order grows time'); ok(P.costRatios({ depth: 8 }).graphBytes > P.costRatios({ depth: 2 }).graphBytes, 'depth grows graph memory');
});
t('equations list covers 8 residual terms', () => { eq(P.EQUATIONS.length, 8); ok(P.EQUATIONS[0].name === 'Continuity', 'first'); });
console.log(`\n${pass} passed, ${fail} failed`); process.exit(fail ? 1 : 0);
