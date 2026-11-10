// physics_cost.js - stage 11 (autograd-cost): the full physics autograd graph, its stored intermediates as an extra MEMORY block
// (memory API setExtraBlock) and graph build+traverse as extra slabs in a step-time bar vs plain supervised. MODEL ESTIMATES.
import * as PM from './physics_math.js';
import { createPhysicsGraph } from './physics_graph.js';
import { frameBox } from './physics_cam.js';

const XS0 = -2.6, ROW_Z = [8.9, 10.6], SLAB_H = 0.8, SLAB_D = 1.2, FULL = 10.5;
const SLAB_COL = [0x3ab795, 0xf08a24, 0x9b6bd6, 0xe0408a, 0xff8cbf];
const SLAB_NAME = ['forward', 'backward', 'Adam', 'graph build', 'graph traverse'];

export function createCost(ctx, shared) {
  const { THREE, util, knobs, model, hud, camera, theme } = ctx, { clamp, smoothstep: ss } = util;
  const group = new THREE.Group(); group.name = 'autograd-cost'; group.visible = false;
  const graph = createPhysicsGraph(ctx, { variant: 'physics', width: 8 }); graph.group.visible = false;
  const gHold = new THREE.Group(); gHold.add(graph.group); gHold.scale.setScalar(0.3); group.add(gHold);
  const boxGeo = new THREE.BoxGeometry(1, 1, 1);
  const slabMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5, metalness: 0.05, transparent: true, opacity: 0.95, emissive: 0xffffff, emissiveIntensity: 0.2 });
  const slabs = new THREE.InstancedMesh(boxGeo, slabMat, 10); slabs.frustumCulled = false; slabs.renderOrder = 4; slabs.setColorAt(0, new THREE.Color());
  for (let i = 0; i < 10; i++) { slabs.instanceMatrix.array[i * 16 + 15] = 1; slabs.setColorAt(i, new THREE.Color(SLAB_COL[i % 5])); }
  const rail = new THREE.Mesh(boxGeo, new THREE.MeshBasicMaterial({ color: 0x26314a, transparent: true, opacity: 0.6 })); group.add(slabs, rail);
  const mk = (t, c, s = 0.32) => { const l = util.makeLabel(THREE, t, { color: c, size: s }); l.visible = false; group.add(l); return l; };
  const rowLbl = [mk('plain supervised step', '#cfe3ff', 0.46), mk('physics-loss step', '#ffd0e4', 0.46)];
  const slabLbl = SLAB_NAME.map((n, i) => mk(n, '#' + SLAB_COL[i].toString(16).padStart(6, '0'), 0.34));
  const ratioLbl = mk('', '#ff9ac4', 0.7), gLbl = mk('autograd graph (stored intermediates)', '#ffd0e4', 0.42);
  const S = { key: '', ext: -1, held: false, cur: new Float32Array(10), tgt: new Float32Array(10), vis: 0, anchor: new THREE.Vector3(4.2, -2.5, 3.5), slabs: null, ratios: null, off: null, ver: 0 };
  const tmpM = new THREE.Matrix4(), q = new THREE.Quaternion(), pos = new THREE.Vector3(), scl = new THREE.Vector3();
  const mem = () => shared.mem();
  const sub = () => { const m = mem(); if (m && !S.off) S.off = m.onLayout(() => { const a = m.getBlockAnchor('e0'); if (a) S.anchor.copy(a); }); };

  function recompute(cfg) {
    const R = PM.costRatios(cfg), sup = R.sup, phy = R.phys; S.ratios = R; S.slabs = { sup, phy }; S.ver++;
    const k = FULL / phy.total;
    [sup, phy].forEach((s, r) => [s.fwd, s.bwd, s.opt, s.build, s.traverse].forEach((t, i) => { S.tgt[r * 5 + i] = t * k; }));
    graph.rebuild(cfg);
    const m = mem(); if (m && S.held) m.setExtraBlock('autograd', { label: 'AUTOGRAD GRAPH', bytes: Math.max(1, R.graphBytes), color: theme.COLORS.physics });
  }

  function update(dt, state, cur) {
    const idx = shared.idx.cost, r = util.revealOf(state, 'autograd.cost'), rg = util.revealOf(state, 'autograd.graph'), after = state.stageIndex > idx;
    const here = state.stageIndex === idx, m = mem(); sub();
    if (here && !S.held) { S.held = true; S.key = ''; m?.hold('physics', true); }
    if (state.stageIndex < idx && S.held) { S.held = false; m?.hold('physics', false); m?.setExtraBlock('autograd', null); }
    S.vis += ((after ? 0 : 1) - S.vis) * (state.reducedMotion ? 1 : 1 - Math.exp(-dt * 4)); if (Math.abs(S.vis - (after ? 0 : 1)) < 0.002) S.vis = after ? 0 : 1;
    const G = r * S.vis, Gg = rg * S.vis; group.visible = G > 0.001 || Gg > 0.001;
    if (!group.visible) return;
    const cfg = ctx.params(), key = [cfg.batch, cfg.width, cfg.depth, cfg.precision, cfg.pdeOrder, cfg.nResiduals, cfg.bandwidthGBs, cfg.peakTflops].join('|');
    if (key !== S.key && (here || !S.slabs)) { S.key = key; recompute(cfg); const a = m?.getBlockAnchor('e0'); if (a) S.anchor.copy(a); }
    // graph above the autograd memory block
    graph.setAlpha(Gg); gHold.visible = Gg > 0.002;
    { const a = m?.getBlockAnchor('e0'); if (a) S.anchor.copy(a); }
    gHold.position.copy(S.anchor).add(pos.set(0, 1.55, 0)); if (cur || here) graph.animate(dt, { speed: 6, pause: 1.0 });
    gLbl.visible = Gg > 0.05; gLbl.material.opacity = Gg; gLbl.position.set(S.anchor.x, S.anchor.y + 3.6, S.anchor.z);
    // slabs
    const floor = m?.FLOOR_Y ?? -5.6, e = state.reducedMotion ? 1 : 1 - Math.exp(-dt * 6);
    for (let i = 0; i < 10; i++) S.cur[i] += (S.tgt[i] - S.cur[i]) * e;
    const mob = innerWidth <= 640;
    for (let row = 0; row < 2; row++) {
      let x = XS0;
      for (let i = 0; i < 5; i++) {
        const len = Math.max(S.cur[row * 5 + i] * G, 1e-4);
        pos.set(x + len / 2, floor + SLAB_H / 2, ROW_Z[row]); scl.set(len, SLAB_H * G + 1e-5, SLAB_D * G + 1e-5); slabs.setMatrixAt(row * 5 + i, tmpM.compose(pos, q.identity(), scl));
        if (row === 1) { const l = slabLbl[i]; l.visible = G > 0.4 && len > 0.05 && !(mob && i % 2 === 0); l.material.opacity = G; l.position.set(x + len / 2, floor + 1.25 + (i % 2) * 0.5, ROW_Z[1] + 0.2); }
        x += len;
      }
      const l = rowLbl[row]; l.visible = G > 0.3; l.material.opacity = G; l.position.set(XS0 + 2.2, floor + (mob && row === 0 ? -0.25 : 1.15), ROW_Z[row] - 0.2);
      if (row === 1) { ratioLbl.visible = G > 0.4; ratioLbl.material.opacity = G; if (mob) ratioLbl.position.set((XS0 + x) / 2, floor + 2.7, ROW_Z[1]); else ratioLbl.position.set(x + 1.6, floor + 0.6, ROW_Z[1]); }
    }
    slabs.instanceMatrix.needsUpdate = true; slabMat.opacity = 0.95 * G;
    rail.visible = false;
    const R = S.ratios;
    if (R) util.setLabel(ratioLbl, `×${R.time.toFixed(1)} time`);
    if (cur && R) {
      const fb = model.fmtBytes, ft = model.fmtTime;
      hud.readout('ratio', 'Step time: physics / plain', `×${R.time.toFixed(1)}  (${ft(R.phys.total)} vs ${ft(R.sup.total)})`, 'model estimate');
      hud.readout('over', 'Graph build + traverse', `${ft(R.phys.build + R.phys.traverse)} vs ${ft(R.sup.build + R.sup.traverse)}`, 'model estimate');
      hud.readout('gm', 'Graph memory (stored)', `+${fb(R.graphBytes)}  (${(R.graphBytes / R.sup.stepCost.memory.total).toFixed(1)}x plain total)`, 'model estimate');
      hud.readout('gn', 'Graph nodes (traversals)', `${R.phys.graph.totalNodes} vs ${R.sup.graph.totalNodes}`, 'model estimate');
    }
  }
  function frame() {
    const mob = innerWidth <= 640;
    return frameBox(THREE, camera, mob ? [-2.6, 8.6, -6.6, -0.6, -3.0, 11.0] : [-3.2, 9.0, -6.6, 1.2, -3.0, 11.8], { az: mob ? -40 : -22, el: mob ? 30 : 28 });
  }
  const api = { get ratios() { return S.ratios; }, get graph() { return graph; }, graphStats() { return { nodes: graph.stats.nodes, edges: graph.stats.edges }; }, get slabTargets() { return Array.from(S.tgt); }, get version() { return S.ver; } };
  return { group, update, frame, enter() { S.key = ''; }, api,
    dispose() { S.off?.(); const m = mem(); if (S.held) { m?.hold('physics', false); m?.setExtraBlock('autograd', null); } graph.dispose(); boxGeo.dispose(); slabMat.dispose(); slabs.dispose(); rail.material.dispose(); util.disposeObject(group); } };
}
