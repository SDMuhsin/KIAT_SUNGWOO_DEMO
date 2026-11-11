// physics_grad.js - stage 10 (phys-grad): SIDE-BY-SIDE autograd graphs, same scale, same camera, same wavefront speed.
//   A: supervised weight gradient dL/dtheta = one reverse sweep.   B: physics-loss gradient = backward through the derivative graph.
// Node/edge counts come from physics_math.buildGraph (nodes == model.autogradGraph). ILLUSTRATIVE layout, MODEL ESTIMATE counts.
import { createPhysicsGraph } from './physics_graph.js';
import { makeControls } from './physics_ui.js';
import { frameBox } from './physics_cam.js';

export const ZONE = { x: 4, y: 22, z: 0 };   // directly above the net / derivative tower: stage 9 -> 13 is a vertical climb through the same world
const SPEED = 6, END_PAD = 7;

export function createGrad(ctx, shared) {
  const { THREE, util, knobs, model, hud, camera } = ctx, { clamp } = util;
  const group = new THREE.Group(); group.name = 'physgrad'; group.position.set(ZONE.x, ZONE.y, ZONE.z); group.visible = false;
  const gA = createPhysicsGraph(ctx, { variant: 'supervised', width: 9 }), gB = createPhysicsGraph(ctx, { variant: 'physics', width: 9 });
  const holdA = new THREE.Group(), holdB = new THREE.Group(); holdA.add(gA.group); holdB.add(gB.group); group.add(holdA, holdB);
  const mk = (t, c, s = 0.5) => { const l = util.makeLabel(THREE, t, { color: c, size: s }); group.add(l); return l; };
  const L = {
    aT: mk('A   Weight gradient', '#6fb2ff', 0.62), aF: mk('dL / dθ   one reverse sweep', '#cfe3ff', 0.46), aC: mk('', '#8fb4ff', 0.4),
    bT: mk('B   Physics-loss gradient', '#ff6fae', 0.62), bF: mk('d/dθ ( dû/dx )   derivative of a derivative', '#ffd0e4', 0.46), bC: mk('', '#ff9ac4', 0.4),
    bN1: mk('entangled with weights + activations', '#ffd0e4', 0.5), bN2: mk('no static graph: rebuilt every step', '#ffd0e4', 0.5),
    aFw: mk('forward (stored)', '#9aa7bd', 0.36), aBw: mk('backward sweep', '#f08a24', 0.36), bFw: mk('forward (stored)', '#9aa7bd', 0.36), bBw: mk('backward sweep', '#f08a24', 0.36),
    aDone: mk('done', '#3ab795', 0.5), bBusy: mk('', '#ff9ac4', 0.5),
  };
  gA.setEdgeStyle(3.4, 1.7); gB.setEdgeStyle(2.1, 1.7);
  // big high-contrast caption (DOM) so the A-vs-B contrast reads in seconds on a projector
  const big = document.createElement('div'); big.id = 'phys-big-caption'; big.setAttribute('aria-live', 'polite');
  Object.assign(big.style, { position: 'fixed', left: '50%', transform: 'translateX(-50%)', zIndex: 4, display: 'none', pointerEvents: 'none', textAlign: 'center', font: '800 clamp(12px, 2.5vw, 30px)/1.15 system-ui,Segoe UI,sans-serif', background: 'rgba(5,8,14,.88)', border: '2px solid #e8edf5', borderRadius: '12px', color: '#fff', maxWidth: 'calc(100vw - 24px)', width: 'max-content', whiteSpace: 'normal', padding: '6px 14px' });
  document.body.appendChild(big);
  const placeBig = () => { const r = document.getElementById('hud-bottom')?.getBoundingClientRect(); big.style.bottom = Math.round(innerHeight - (r ? r.top : innerHeight - 170) + 8) + 'px'; };
  const ui = makeControls('phys-controls-grad', 'Backward sweep animation', '#f08a24');
  const S = { dirty: true, h: -2, hold: 0, playing: true, vis: 0, portrait: false, ver: 0, cfgKey: '' };
  const maxH = () => Math.max(gA.maxHops, gB.maxHops) + END_PAD;
  ui.replay.onclick = () => { S.h = -2; S.hold = 0; S.playing = true; syncUI(); };
  ui.play.onclick = () => { S.playing = !S.playing; syncUI(); };
  ui.scrub.oninput = () => { S.playing = false; S.h = ui.scrub.value / 1000 * maxH(); syncUI(); };
  function syncUI() { ui.play.textContent = S.playing ? '❚❚ Pause' : '▶ Play'; if (document.activeElement !== ui.scrub) ui.scrub.value = Math.round(clamp(S.h / maxH()) * 1000); ui.note.textContent = 'same speed'; }
  const timer = setInterval(() => { if (ui.el.style.display !== 'none') syncUI(); }, 200);
  for (const k of ['depth', 'pdeOrder', 'nResiduals']) knobs.on(k, () => { S.dirty = true; });

  function layout() {
    const port = innerWidth < innerHeight * 0.85; S.portrait = port;
    const bA = gA.stats.bounds, bB = gB.stats.bounds; if (!bA || !bB) return;
    const topB = bB.h - bB.H, topA = bA.h - bA.H;   // local y of the forward block's top
    const P = (l, x, y, z = 0) => l.position.set(x, y, z);
    if (!port) {
      const dyA = topB - topA, xa = -5.6, xb = 5.6;
      holdA.position.set(xa, dyA, 0); holdB.position.set(xb, 0, 0);
      P(L.aT, xa, topB + 1.75); P(L.aF, xa, topB + 1.15); P(L.bT, xb, topB + 1.75); P(L.bF, xb, topB + 1.15);
      P(L.aFw, xa, topB + 0.5); P(L.bFw, xb, topB + 0.5);
      P(L.aBw, xa, dyA - bA.H - 0.5); P(L.bBw, xb, -bB.H - 0.6);
      P(L.aC, xa, dyA - bA.H - 1.1); P(L.bC, xb, -bB.H - 1.2);
      P(L.aDone, xa, dyA - bA.H - 1.8); P(L.bN1, xb, -bB.H - 1.85); P(L.bN2, xb, -bB.H - 2.4); P(L.bBusy, xb, -bB.H - 2.95);
    } else {
      const yA = 4.6, yB = -3.4;
      holdA.position.set(0, yA, 0); holdB.position.set(0, yB, 0);
      P(L.aT, 0, yA + topA + 1.9); P(L.aF, 0, yA + topA + 1.2); P(L.aFw, 0, yA + topA + 0.45); P(L.aBw, 0, yA - bA.H - 0.5); P(L.aC, 0, yA - bA.H - 1.1); P(L.aDone, 3.1, yA - bA.H - 0.5);
      const tB = yB + topB;
      P(L.bT, 0, tB + 2.1); P(L.bF, 0, tB + 1.4); P(L.bFw, 0, tB + 0.55); P(L.bBw, 0, yB - bB.H - 0.65);
      P(L.bC, 0, yB - bB.H - 1.3); P(L.bN1, 0, yB - bB.H - 2.0); P(L.bN2, 0, yB - bB.H - 2.65); P(L.bBusy, 3.1, yB - bB.H - 0.65);
    }
  }
  function rebuild() {
    const cfg = ctx.params(); S.dirty = false; S.ver++;
    gA.rebuild(cfg); gB.rebuild(cfg); layout();
  }
  function fmt() {
    util.setLabel(L.aC, `${gA.stats.nodes} nodes · ${gA.stats.edges} edges · ${gA.stats.hops + 1} steps`);
    util.setLabel(L.bC, `${gB.stats.nodes} nodes · ${gB.stats.edges} edges · ${gB.stats.hops + 1} steps`);
  }
  function updateBig(show) {
    big.style.display = show ? 'block' : 'none'; if (!show) return; placeBig();
    const a = gA.stats.nodes, b = gB.stats.nodes;
    big.innerHTML = `<span style="color:#7fb8ff">Weights: ${a} nodes</span> &nbsp;vs&nbsp; <span style="color:#ff7eb6">Physics: ${b} nodes</span> &nbsp;<span style="color:#ffd24a">&times;${(b / Math.max(1, a)).toFixed(1)} more</span>`;
  }
  function update(dt, state, cur) {
    const idx = shared.idx.physgrad, r = util.revealOf(state, 'physgrad.weightpath'), rp = util.revealOf(state, 'physgrad.physpath'), after = state.stageIndex > idx;
    S.vis += ((after ? 0 : 1) - S.vis) * (state.reducedMotion ? 1 : 1 - Math.exp(-dt * 4)); if (Math.abs(S.vis - (after ? 0 : 1)) < 0.002) S.vis = after ? 0 : 1;
    const G = Math.max(r, rp) * S.vis; group.visible = G > 0.001; ui.show(cur && G > 0.5); if (innerWidth > 900) { const r = document.getElementById('hud-top').getBoundingClientRect(); ui.el.style.top = '12px'; ui.el.style.left = Math.round(r.right + 8) + 'px'; } updateBig(cur && G > 0.5 && !!gB.stats.nodes);
    if (!group.visible) return;
    if (S.dirty) { rebuild(); fmt(); }
    if (S.portrait !== (innerWidth < innerHeight * 0.85)) layout();
    const ga = r * S.vis, gb = rp * S.vis;
    gA.setAlpha(ga); gB.setAlpha(gb);
    if (cur) {
      if (state.reducedMotion) S.h = maxH() - 0.5;
      else if (S.playing) { if (S.h < maxH()) S.h += dt * SPEED; else { S.hold += dt; if (S.hold > 1.4) { S.hold = 0; S.h = -2; } } }
    } else if (after) S.h = maxH();
    gA.setSweep(S.h); gB.setSweep(S.h);
    const aDone = S.h >= gA.maxHops, bDone = S.h >= gB.maxHops;
    const fade = (l, a) => { l.visible = a > 0.02; l.material.opacity = clamp(a); };
    for (const k of ['aT', 'aF', 'aC']) fade(L[k], ga); for (const k of ['bT', 'bF', 'bC', 'bN1', 'bN2']) fade(L[k], gb); for (const k of ['aFw', 'aBw', 'bFw', 'bBw']) L[k].visible = false;
    fade(L.aDone, ga * (aDone ? 1 : 0)); fade(L.bBusy, gb);
    util.setLabel(L.bBusy, bDone ? 'done' : `still going... step ${clamp(Math.floor(S.h) + 1, 0, gB.maxHops + 1)}`);
    L.bBusy.material.color?.set?.(bDone ? 0x3ab795 : 0xffffff);
    if (state.stageId === 'phys-grad') {
      const cfg = ctx.params(), a = model.autogradGraph(cfg, 'supervised'), b = model.autogradGraph(cfg, 'physics');
      hud.readout('nodes', 'Autograd nodes  A vs B', `${a.totalNodes} vs ${b.totalNodes}  (x${(b.totalNodes / a.totalNodes).toFixed(1)})`, 'model estimate');
      hud.readout('edges', 'Edges  A vs B', `${gA.stats.edges} vs ${gB.stats.edges}`, 'illustrative');
      hud.readout('trav', 'Passes over the net  A vs B', `${a.passes} vs ${b.passes}`, 'model estimate');
      hud.readout('hops', 'Backward steps  A vs B', `${gA.stats.hops + 1} vs ${gB.stats.hops + 1}`, 'illustrative');
    }
  }
  function frame() {
    if (S.dirty) { rebuild(); fmt(); }
    const mob = innerWidth < innerHeight * 0.85, c = ZONE, bA = gA.stats.bounds, bB = gB.stats.bounds;
    if (mob) {
      const yTop = 4.6 + bA.h - bA.H + 2.7, yBot = -3.4 - bB.H - 3.2;
      return frameBox(THREE, camera, [c.x - 4.5, c.x + 4.5, yBot + c.y, yTop + c.y, -2.2, 2.2], { az: -14, el: 16, reserve: { top: 72, bottom: 40 } });
    }
    return frameBox(THREE, camera, [c.x - 10.2, c.x + 10.2, -bB.H - 3.0 + c.y, bB.h - bB.H + 2.2 + c.y, -2.4, 2.4], { az: -12, el: 24, reserve: { top: -26, bottom: 24 } });
  }
  const api = {
    graphStats() { return { supervised: { nodes: gA.stats.nodes, edges: gA.stats.edges, hops: gA.stats.hops }, physics: { nodes: gB.stats.nodes, edges: gB.stats.edges, hops: gB.stats.hops }, version: S.ver }; },
    get sweep() { return S.h; }, setSweep(h) { S.playing = false; S.h = h; }, graphs: { supervised: gA, physics: gB },
  };
  rebuild(); fmt();
  return { group, update, frame, enter() { S.h = ctx.state.reducedMotion ? 1e3 : -2; S.hold = 0; S.playing = true; S.dirty = true; }, api,
    dispose() { clearInterval(timer); big.remove(); ui.dispose(); gA.dispose(); gB.dispose(); util.disposeObject(group); } };
}
