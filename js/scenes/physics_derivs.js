// physics_derivs.js - stage 9 (derivs): differentiating the network w.r.t. its INPUTS spawns a tower of derivative streams
// through every layer; the model.js multiplier is shown as growing slabs. ILLUSTRATIVE layout, MODEL ESTIMATE numbers.
import * as PM from './physics_math.js';
import { createBars } from './physics_graph.js';
import { makeControls } from './physics_ui.js';
import { frameBox } from './physics_cam.js';

const LY0 = 3.0, LDY = 1.3, ZS = 0.3, MAXS = 43, MAXLAYER = 14, T_END = 6.5;
const ORDER_COL = [0x3ab795, 0xf2c14e, 0xe0408a, 0xff9ac4];
const PILE = { x: -3.3, y: 2.0, z: 0, u: 0.11, w: 1.1 };

export function createDerivs(ctx, shared) {
  const { THREE, util, knobs, model, hud, camera } = ctx, { clamp, smoothstep: ss } = util;
  const group = new THREE.Group(); group.name = 'derivs'; group.visible = false;
  const bars = createBars(THREE, MAXS * MAXLAYER * 2 + 80, 0.9), links = createBars(THREE, MAXS * MAXLAYER + 40, 0.35);
  bars.mesh.renderOrder = 4; links.mesh.renderOrder = 3;
  const sph = new THREE.SphereGeometry(1, 10, 8);
  const nodeMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95 });
  const nodes = new THREE.InstancedMesh(sph, nodeMat, MAXS * MAXLAYER + 16); nodes.frustumCulled = false; nodes.renderOrder = 5; nodes.setColorAt(0, new THREE.Color());
  const dots = new THREE.InstancedMesh(sph, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 1 }), MAXS * 2); dots.frustumCulled = false; dots.renderOrder = 6; dots.setColorAt(0, new THREE.Color());
  const boxGeo = new THREE.BoxGeometry(1, 1, 1);
  const slabMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5, metalness: 0.05, transparent: true, opacity: 0.92, emissive: 0xffffff, emissiveIntensity: 0.2 });
  const slabs = new THREE.InstancedMesh(boxGeo, slabMat, 4); slabs.frustumCulled = false; slabs.renderOrder = 4; slabs.setColorAt(0, new THREE.Color());
  for (let i = 0; i < 4; i++) { slabs.instanceMatrix.array[i * 16 + 15] = 1; slabs.setColorAt(i, new THREE.Color(ORDER_COL[i])); }
  group.add(bars.mesh, links.mesh, nodes, dots, slabs);
  const mk = (t, c, s = 0.3) => { const l = util.makeLabel(THREE, t, { color: c, size: s }); l.visible = false; group.add(l); return l; };
  const laneLbl = [0, 1, 2, 3].map((k) => mk(PM.ORDER_NAMES[k], '#' + ORDER_COL[k].toString(16).padStart(6, '0'), 0.34));
  const slabLbl = [0, 1, 2, 3].map(() => mk('x', '#e8edf5', 0.27));
  const totLbl = mk('x', '#ff9ac4', 0.34), resLbl = mk('residual terms', '#ff9ac4', 0.3), capLbl = mk('derivative of the network w.r.t. its INPUTS', '#e8edf5', 0.3);
  const resGeo = boxGeo, resMesh = new THREE.InstancedMesh(resGeo, new THREE.MeshBasicMaterial({ color: 0xe0408a, transparent: true, opacity: 0.95 }), 8); resMesh.frustumCulled = false; group.add(resMesh);

  const ui = makeControls('phys-controls-derivs', 'Derivative tower animation', '#f2c14e'); ui.scrub.style.display = 'none'; ui.play.style.display = 'none';
  const S = { dirty: true, tt: 0, vis: 0, lanes: [], counts: [1], T: 1, streams: [], nl: 0, resPos: [], ver: 0 };
  ui.replay.onclick = () => { S.tt = 0; ui.note.textContent = 'building'; };
  knobs.on('pdeOrder', () => { S.dirty = true; }); knobs.on('depth', () => { S.dirty = true; }); knobs.on('nResiduals', () => { S.dirty = true; });
  let off = null; const sub = () => { const n = shared.net(); if (n && !off) off = n.onRebuild(() => { S.dirty = true; }); }; 

  const WHITE = new THREE.Color(0xffffff), A = new THREE.Vector3(), B = new THREE.Vector3(), tmp = new THREE.Matrix4(), q = new THREE.Quaternion(), pos = new THREE.Vector3(), scl = new THREE.Vector3(), col = new THREE.Color();
  function rebuild() {
    const net = shared.net(); if (!net) return; sub(); S.dirty = false; S.ver++;
    const info = net.getLayerInfo(), nl = Math.min(info.length, MAXLAYER), cfg = ctx.params();
    const counts = PM.branchCounts(cfg); S.counts = counts; S.T = counts.reduce((a, x) => a + x, 0); S.nl = nl;
    const xs = info.slice(0, nl).map((l) => l.x);
    const streams = []; let nb = 0, nn = 0, nlk = 0;
    counts.forEach((cnt, k) => {
      for (let j = 0; j < cnt; j++) {
        const z = (j - (cnt - 1) / 2) * ZS, y = LY0 + k * LDY, st = { k, j, z, y, parent: -1 };
        if (k > 0) { const off0 = counts.slice(0, k - 1).reduce((a, x) => a + x, 0); st.parent = off0 + (j % counts[k - 1]); }
        streams.push(st);
        const c = ORDER_COL[k];
        for (let l = 0; l < nl; l++) {
          A.set(xs[l], y, z);
          if (l < nl - 1) { B.set(xs[l + 1], y, z); bars.set(nb++, A, B, 0.05, c); }
          nodes.setColorAt(nn, col.set(c)); pos.copy(A); scl.setScalar(0.08); nodes.setMatrixAt(nn++, tmp.compose(pos, q.identity(), scl));
          const pz = k > 0 ? streams[st.parent].z : z, py = k > 0 ? streams[st.parent].y : 1.05;
          if (k > 0 || l < nl) { B.set(xs[l], py, pz); links.set(nlk++, A, B, 0.012, c); }
        }
      }
    });
    bars.commit(nb); links.commit(nlk); nodes.count = nn; nodes.instanceMatrix.needsUpdate = true; if (nodes.instanceColor) nodes.instanceColor.needsUpdate = true;
    S.streams = streams; S.xs = xs;
    // residual assembly cubes at the output end
    const nr = knobs.get('nResiduals'), xe = xs[nl - 1] + 0.9; S.resPos = [];
    for (let i = 0; i < 8; i++) { const y = LY0 + 0.4 + (i - (nr - 1) / 2) * 0.28 + ((counts.length - 1) * LDY) / 2; S.resPos.push([xe, y, 0]); }
    laneLbl.forEach((l, k) => { l.position.set(xs[0] + 0.3 + (k % 2) * 0.0, LY0 + k * LDY + 0.4, ((counts[k] - 1) / 2) * ZS + 0.9); l.userData.on = k < counts.length; });
    resLbl.position.set(xe, LY0 + 0.4 + ((counts.length - 1) * LDY) / 2 + nr * 0.16 + 0.35, 0);
    capLbl.position.set((xs[0] + xs[nl - 1]) / 2, LY0 - 0.6, 3.8);
  }

  function update(dt, state, cur) {
    const idx = shared.idx.derivs, r = util.revealOf(state, 'derivs.tower'), after = state.stageIndex > idx;
    S.vis += ((after ? 0 : 1) - S.vis) * (state.reducedMotion ? 1 : 1 - Math.exp(-dt * 4)); if (Math.abs(S.vis - (after ? 0 : 1)) < 0.002) S.vis = after ? 0 : 1;
    const G = r * S.vis; group.visible = G > 0.001; ui.show(cur && r > 0.5);
    if (!group.visible) return;
    if (S.dirty) rebuild();
    if (cur) S.tt = state.reducedMotion ? T_END : Math.min(T_END + 100, S.tt + dt);
    else if (after) S.tt = T_END;
    const cfg = ctx.params(), tt = S.tt, counts = S.counts, nl = S.nl, T = S.T;
    bars.mesh.material.opacity = 0.9 * G; links.mesh.material.opacity = 0.35 * G; nodeMat.opacity = 0.95 * G; slabMat.opacity = 0.92 * G; dots.material.opacity = G; resMesh.material.opacity = 0.95 * G;
    // bars/nodes/links are laid out lane by lane and lanes reveal sequentially, so showing a prefix of instances reveals lane k
    let nbShown = 0, nnShown = 0, nlShown = 0;
    const gate = counts.map((_, k) => ss(0.7 + k * 1.3, 1.5 + k * 1.3, tt));
    counts.forEach((cnt, k) => { const gk = gate[k]; nbShown += Math.round(cnt * (nl - 1) * gk); nnShown += Math.round(cnt * nl * gk); nlShown += Math.round(cnt * nl * gk); });
    const nbAll = T * (nl - 1), nnAll = T * nl;
    bars.mesh.count = clamp(nbShown, 0, nbAll); nodes.count = clamp(nnShown, 0, nnAll); links.mesh.count = clamp(nlShown, 0, nnAll);
    // moving dots: 2 per revealed stream, speed constant, phase staggered
    let di = 0;
    const xs = S.xs ?? [], span = xs.length ? xs[nl - 1] - xs[0] : 1;
    for (const st of S.streams) {
      const gk = gate[st.k];
      for (let h = 0; h < 2; h++) {
        const ph = ((state.time * 0.22 + h * 0.5 + st.j * 0.07 + st.k * 0.13) % 1);
        pos.set(xs[0] + ph * span, st.y, st.z); scl.setScalar(0.1 * gk * G + 1e-5);
        dots.setMatrixAt(di, tmp.compose(pos, q.identity(), scl)); dots.setColorAt(di, col.set(ORDER_COL[st.k]).lerp(WHITE, 0.5)); di++;
      }
    }
    dots.count = di; dots.instanceMatrix.needsUpdate = true; dots.instanceColor.needsUpdate = true;
    // slabs (growing blocks: units per order from model.towerFactor)
    let y = PILE.y;
    for (let k = 0; k < 4; k++) {
      const n = counts[k] ?? 0, h = n * PILE.u * (k < counts.length ? gate[k] : 0);
      pos.set(PILE.x, y + h / 2, PILE.z); scl.set(PILE.w * G + 1e-5, Math.max(h * G, 1e-5), PILE.w * G + 1e-5); slabs.setMatrixAt(k, tmp.compose(pos, q.identity(), scl));
      const lb = slabLbl[k], on = k < counts.length && gate[k] > 0.3 && h > 0.01;
      lb.visible = on; lb.material.opacity = G; if (on) { util.setLabel(lb, (k === 0 ? 'u: x1' : `${k === 1 ? '1st' : k === 2 ? '2nd' : '3rd'} order: +x${n}`)); lb.position.set(PILE.x + PILE.w / 2 + 1.0, y + h / 2, PILE.z); }
      y += h;
    }
    slabs.instanceMatrix.needsUpdate = true;
    totLbl.visible = G > 0.3; totLbl.material.opacity = G; util.setLabel(totLbl, `x${model.towerFactor(cfg)} compute & activations`); totLbl.position.set(PILE.x, y + 0.45, PILE.z);
    laneLbl.forEach((l, k) => { const on = l.userData.on && gate[k] > 0.3; l.visible = on; l.material.opacity = G * gate[k]; });
    capLbl.visible = G > 0.3; capLbl.material.opacity = G;
    // residual cubes
    const nr = knobs.get('nResiduals'), gr = ss(0.7 + (counts.length) * 1.3, 1.6 + counts.length * 1.3, tt);
    for (let i = 0; i < 8; i++) { const p = S.resPos[i] ?? [0, 0, 0]; pos.set(p[0], p[1], p[2]); const s = (i < nr ? 0.16 : 0) * gr * G + 1e-5; scl.setScalar(s); resMesh.setMatrixAt(i, tmp.compose(pos, q.identity(), scl)); }
    resMesh.instanceMatrix.needsUpdate = true; resLbl.visible = gr > 0.3; resLbl.material.opacity = G * gr;
    ui.note.textContent = tt < 1.6 + counts.length * 1.3 ? 'building tower' : 'complete';
    if (cur) {
      const g = model.autogradGraph(cfg, 'physics'), s = model.autogradGraph(cfg, 'supervised');
      hud.readout('T', 'Compute & activation multiplier', `x${model.towerFactor(cfg)}`, 'model estimate');
      hud.readout('br', 'Derivative branches', `${T} (orders: ${counts.join(' + ')})`, 'model estimate');
      hud.readout('gn', 'Graph nodes', `${g.forwardNodes} vs ${s.forwardNodes} plain`, 'model estimate');
    }
  }
  function frame() {
    const mob = innerWidth <= 640, zb = mob ? 3 : 4.6;
    return frameBox(THREE, camera, [-4.6, 8.8, -1.0, 6.9, -zb, zb], { az: mob ? -40 : -26, el: mob ? 22 : 14, reserve: { top: 44 } });
  }
  const api = { get tower() { return { counts: S.counts, T: S.T, streams: S.streams.length, layers: S.nl, version: S.ver }; }, get clock() { return S.tt; }, setClock(t) { S.tt = t; } };
  return { group, update, frame, enter() { S.tt = ctx.state.reducedMotion ? T_END : 0; S.dirty = true; }, api,
    dispose() { off?.(); ui.dispose(); bars.dispose(); links.dispose(); sph.dispose(); boxGeo.dispose(); nodeMat.dispose(); slabMat.dispose(); nodes.dispose(); dots.dispose(); slabs.dispose(); resMesh.dispose(); util.disposeObject(group); } };
}
