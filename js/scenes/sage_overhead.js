// sage_overhead.js - stage 13 (sage-overhead): BEFORE/AFTER side by side at the same scale. Left: the autograd graph of the physics loss
// (physics_graph.js, variant 'physics'); right: SAGE = a thin closed-form "v" lane + ONE clean network backward (variant 'supervised').
// The blend (knob sageBlend 0..1, 0 = PyTorch autograd, 1 = SAGE) drives graph.setCollapse(blend), the step-time bar, the AD-graph memory
// cube (and the memory scene's extra block via sage.js), and the matmul tiles 24 -> 10. All numbers: sage_math.overheadState = MODEL ESTIMATES.
import * as SM from './sage_math.js';
import { ZONES, createBoxes, createLabels, useStyle } from './sage_kit.js';
import { frameBox } from './physics_cam.js';
import { makeControls } from './physics_ui.js';
import { createPhysicsGraph } from './physics_graph.js';

const Z = ZONES.overhead, SLAB_COL = [0x3ab795, 0xf08a24, 0x9b6bd6, 0xe0408a, 0xff8cbf], SLAB_NAME = ['forward', 'backward', 'Adam', 'graph build', 'graph traverse'];
const COL = { sage: 0x2fd0e6, phys: 0xe0408a, dim: 0x1b2438 };

export function createOverhead(ctx, shared) {
  const { THREE, util, camera, knobs, hud, model } = ctx, { smoothstep: ss, clamp } = util;
  const group = new THREE.Group(); group.name = 'sage-overhead'; group.position.set(...Z); group.visible = false;
  const before = createPhysicsGraph(ctx, { variant: 'physics', width: 8 }), after = createPhysicsGraph(ctx, { variant: 'supervised', width: 8 });
  const gB = new THREE.Group(), gA = new THREE.Group(); gB.add(before.group); gA.add(after.group); group.add(gB, gA);
  const boxes = createBoxes(THREE, 80, { emissive: 0.22 }); group.add(boxes.mesh);
  const L = createLabels(THREE, util, group), restore = useStyle();
  const lB = L.add('BEFORE: autograd graph', '#ffb3d4', 0.55), lA = L.add('AFTER: SAGE', '#9ee9f4', 0.55);
  const lBn = L.add('618 graph nodes', '#ffd0e4', 0.5), lAn = L.add('30 nodes: network only', '#cfeef5', 0.5);
  const lLane = L.add('v: closed-form lane (generated fn), no graph', '#2fd0e6', 0.4), lGone = L.add('graph eliminated', '#8793a6', 0.55);
  const lHead = L.add('', '#f2c14e', 0.7), lBar = L.add('step time', '#e8edf5', 0.4), lBarV = L.add('', '#f2c14e', 0.42), lMem = L.add('AD-graph memory', '#e8edf5', 0.4), lMemV = L.add('', '#f2c14e', 0.42), lMm = L.add('matmuls', '#e8edf5', 0.4), lMmV = L.add('', '#f2c14e', 0.42);
  const segL = SLAB_NAME.map((n, i) => L.add(n, '#' + SLAB_COL[i].toString(16).padStart(6, '0'), 0.3));
  const ctl = makeControls('sage-overhead-controls', 'Before/after blend controls', '#2fd0e6'); ctl.note.style.display = 'none';
  const S = { blend: 0, auto: { on: false, t: 0 }, key: '', mob: null, vis: 0, st: null, lt: 0, setting: false, ver: 0, cy: 0, lastTxt: {}, laneT: 0 };
  const tmp = { b: 0 };

  knobs.define({ id: 'sageBlend', label: 'SAGE off → on', min: 0, max: 1, step: 0.01, value: 0, group: 'SAGE', help: '0 = PyTorch autograd graph for the physics loss, 1 = SAGE (closed-form v + one network backward). Drives the collapse.' });
  const offBlend = knobs.on('sageBlend', (v) => { S.blend = v; if (!S.setting) S.auto.on = false; if (document.activeElement !== ctl.scrub) ctl.scrub.value = Math.round(v * 1000); });
  const setBlend = (v) => { S.setting = true; knobs.set('sageBlend', v); S.setting = false; };
  ctl.replay.onclick = () => { setBlend(0); S.auto.on = true; S.auto.t = 0; ctl.play.textContent = '❚❚ Pause'; };
  ctl.play.onclick = () => { S.auto.on = !S.auto.on; ctl.play.textContent = S.auto.on ? '❚❚ Pause' : '▶ Play'; };
  ctl.scrub.oninput = () => { S.auto.on = false; ctl.play.textContent = '▶ Play'; knobs.set('sageBlend', ctl.scrub.value / 1000); };
  ctl.scrub.setAttribute('aria-label', 'SAGE off to on'); ctl.replay.textContent = '↻ Replay';

  function layout(mob) {
    S.mob = mob; const k = mob ? 1.3 : 1;
    const rs = (sp, base, t) => { sp.userData.opts.size = base * k; sp.userData.text = null; util.setLabel(sp, t); };
    rs(lB, 0.55, 'BEFORE: autograd graph'); rs(lA, 0.55, 'AFTER: SAGE'); rs(lBn, 0.5, S.lastTxt.bn ?? '618 graph nodes'); rs(lAn, 0.5, S.lastTxt.an ?? '30 nodes: network only'); rs(lLane, 0.4, 'v: closed-form lane (generated fn), no graph');
    rs(lGone, 0.55, 'graph eliminated'); rs(lBar, 0.4, 'step time (model estimate)'); rs(lMem, 0.4, 'AD-graph memory'); rs(lMm, 0.4, 'matmuls');
    rs(lBarV, 0.42, S.lastTxt.bar ?? ' '); rs(lMemV, 0.42, S.lastTxt.mem ?? ' '); rs(lMmV, 0.42, S.lastTxt.mm ?? ' ');
    SLAB_NAME.forEach((n, i) => rs(segL[i], 0.3, n));
    S.key = '';
  }
  // geometry of the metric row (desktop / mobile)
  const geo = (mob) => mob
    ? { bx: -5.4, by: -5.0, bw: 5.2, cx: 1.25, cy: -4.7, mx: 3.9, my: -4.9, gb: [0, 6.0], ga: [0, -0.9], laneY: 0.4, tB: 9.5, nB: 2.9, tA: 1.75, nA: -2.45 }
    : { bx: -10.2, by: -5.0, bw: 7.0, cx: 0.9, cy: -4.7, mx: 6.1, my: -4.9, gb: [-5.8, 0.9], ga: [5.8, 0.9], laneY: 2.7, tB: 4.4, nB: -2.55, tA: 4.4, nA: -1.55 };

  function rebuild(cfg) {
    before.rebuild(cfg); after.rebuild(cfg);
    S.ver++; S.cfg = cfg;
  }
  function placeGraphs(mob) {
    const g = geo(mob), bb = before.stats.bounds, ab = after.stats.bounds; if (!bb || !ab) return;
    const cy = (b) => (b.h - 0.7 - 2 * b.H) / 2;
    gB.position.set(g.gb[0], g.gb[1] - cy(bb), 0); gA.position.set(g.ga[0], g.ga[1] - cy(ab), 0);
    S.frame = { cx: g.gb[0], cy: g.gb[1], w: 8.6, h: bb.h + 0.5 };
    lB.position.set(g.gb[0], g.tB, 0.5); lBn.position.set(g.gb[0], Math.min(g.nB, g.gb[1] - S.frame.h / 2 - 0.45), 0.6);
    lA.position.set(g.ga[0], g.tA, 0.5); lAn.position.set(g.ga[0], g.nA, 0.6); lGone.position.set(g.gb[0], g.gb[1], 0.8);
  }

  function update(dt, state, cur) {
    const idx = shared.idx.overhead, r = util.revealOf(state, 'sage.before_after'), here = state.stageIndex === idx;
    const tv = state.stageIndex > idx ? 0 : 1; S.vis += (tv - S.vis) * (state.reducedMotion ? 1 : 1 - Math.exp(-dt * 4)); if (Math.abs(S.vis - tv) < 0.003) S.vis = tv;
    const G = r * S.vis; group.visible = G > 0.002;
    const showDom = here && r > 0.5; ctl.show(showDom);
    if (!group.visible) return;
    const mob = innerWidth <= 640; if (mob !== S.mob) { layout(mob); S.placed = false; }
    ctl.scrub.style.display = mob ? '' : 'none'; ctl.replay.style.display = '';
    // auto-play of the collapse
    if (here && S.auto.on) {
      S.auto.t += dt; const u = state.reducedMotion ? 1 : clamp((S.auto.t - 1.1) / 3.4);
      setBlend(util.easeInOutCubic(u)); if (u >= 1) S.auto.on = false;
    }
    const cfg = ctx.params(), key = [cfg.batch, cfg.width, cfg.depth, cfg.precision, cfg.pdeOrder, cfg.nResiduals, cfg.bandwidthGBs, cfg.peakTflops].join('|');
    if (key !== S.key) { S.key = key; rebuild(cfg); S.placed = false; }
    if (!S.placed) { placeGraphs(mob); S.placed = true; }
    const b = S.blend, os = SM.overheadState(cfg, b); S.st = os; shared.blend = b;
    before.setAlpha(G); after.setAlpha(G * (0.4 + 0.6 * ss(0, 0.7, b))); before.setCollapse(b); 
    before.animate(dt, { speed: 6, pause: 1.0 }); after.animate(dt, { speed: 6, pause: 1.0 });
    // ---------------- metric row + lane
    const g = geo(mob), B = boxes;
    // v lane above the AFTER graph: thin cyan rail + dashes moving toward the network
    const lx0 = g.ga[0] - 4, lx1 = g.ga[0] + 4, ly = g.laneY, laneA = ss(0.0, 0.8, b) * 0.85 + 0.15;
    B.set(0, (lx0 + lx1) / 2, ly, 0, 8 * laneA, 0.06, 0.06, COL.sage);
    S.laneT = (S.laneT + dt * 0.8) % 1;
    for (let i = 0; i < 5; i++) { const u = (S.laneT + i / 5) % 1; B.set(1 + i, lx0 + 8 * u, ly, 0.05, 0.34, 0.17, 0.17, 0xffffff); }
    lLane.position.set(g.ga[0], ly + 0.55, 0.5);
    { const f = S.frame; if (f) { const t = 0.05, c = 0x2a3550; B.set(60, f.cx, f.cy + f.h / 2, -0.2, f.w, t, t, c); B.set(61, f.cx, f.cy - f.h / 2, -0.2, f.w, t, t, c); B.set(62, f.cx - f.w / 2, f.cy, -0.2, t, f.h, t, c); B.set(63, f.cx + f.w / 2, f.cy, -0.2, t, f.h, t, c); } }
    // step bar
    const BW = g.bw, tot0 = os.totalBefore; let x = g.bx;
    B.set(6, g.bx + BW / 2, g.by, -0.2, BW + 0.12, 0.98, 0.12, COL.dim);
    for (let i = 0; i < 5; i++) { const w = os.slabs[i] / tot0 * BW; B.set(7 + i, x + w / 2, g.by, 0, w, 0.7, 0.45, SLAB_COL[i]); x += w; }
    for (let i = 0; i < 5; i++) { const w = os.slabs[i] / tot0 * BW; const show = os.slabsBefore[i] / tot0 > 0.07 && w / BW > 0.045; let xs = g.bx; for (let j = 0; j < i; j++) xs += os.slabs[j] / tot0 * BW; segL[i].position.set(xs + w / 2, g.by + (i % 2 ? -0.75 : 0.72) * (mob ? 1.15 : 1), 0.5); L.show(segL[i], S.mob ? 0 : G * (show || (i === 3 || i === 4) && os.slabsBefore[i] / tot0 > 0.07 ? Math.min(1, w / BW * 40) : 0)); }
    lBar.position.set(g.bx + BW / 2, g.by + (mob ? 1.55 : 1.45), 0.5); lBarV.position.set(g.bx + BW / 2, g.by - (mob ? 1.6 : 1.45), 0.5);
    // AD-graph memory cube (volume ~ bytes), ghost frame at full size
    const E = 1.9, e = E * Math.cbrt(clamp(os.memBytes / Math.max(1, os.memBytesFull)));
    B.set(12, g.cx, g.cy, -0.9, E + 0.1, E + 0.1, 0.05, COL.dim);
    B.set(13, g.cx, g.cy - E / 2 + e / 2, 0, e, e, e, COL.phys);
    lMem.position.set(g.cx, g.cy + E / 2 + 0.75, 0.5); lMemV.position.set(g.cx, g.cy - E / 2 - 0.65, 0.5);
    // matmul tiles 6 x 4
    const mm = SM.lerp(os.matmulsBefore, os.matmulsAfter, b);
    for (let i = 0; i < 24; i++) {
      const cx = i % 6, cr = (i / 6) | 0, tx = g.mx + (cx - 2.5) * 0.46, ty = g.my + (1.5 - cr) * 0.46, a = clamp(mm - i);
      B.set(14 + i, tx, ty, 0, 0.38 * (0.25 + 0.75 * a), 0.38 * (0.25 + 0.75 * a), 0.3, a > 0.5 ? COL.sage : 0x5a2a45);
    }
    lMm.position.set(g.mx, g.my + 1.55, 0.5); lMmV.position.set(g.mx, g.my - 1.5, 0.5);
    B.commit(); B.mat.opacity = G;
    // labels (text throttled to ~8 Hz so no texture churn while scrubbing)
    S.lt += dt;
    const fb = model.fmtBytes, ft = model.fmtTime;
    if (S.lt > 0.12) {
      S.lt = 0; const set = (sp, k, txt) => { if (S.lastTxt[k] !== txt) { S.lastTxt[k] = txt; L.set(sp, txt); } };
      set(lBn, 'bn', `${os.adNodes.toLocaleString('en')} graph nodes`); set(lAn, 'an', `${os.netNodes} nodes: network only`);
      const mb = S.mob; set(lBarV, 'bar', mb ? ft(os.total) : `${ft(os.total)}  (${ft(os.totalBefore)} → ${ft(os.totalAfter)})`); set(lMemV, 'mem', mb ? fb(os.memBytes) : `${fb(os.memBytes)}  (${fb(os.memBytesFull)} → 0)`); set(lMmV, 'mm', mb ? `${Math.round(mm)}` : `${Math.round(mm)}  (${os.matmulsBefore} → ${os.matmulsAfter})`);
    }
    if (S.lt === 0 && !S.mob) { const t = `step \u00f7${(os.totalBefore / os.total).toFixed(1)}  (model)`; if (S.lastTxt.hd !== t) { S.lastTxt.hd = t; L.set(lHead, t); } }
    lHead.position.set(0, 4.45, 0.5);
    const A = G;
    L.show(lHead, S.mob ? 0 : A * ss(0.05, 0.4, b));
    L.show(lB, A * (1 - 0.35 * ss(0.5, 1, b))); L.show(lA, A); L.show(lBn, A); L.show(lAn, A * (0.5 + 0.5 * b)); L.show(lLane, A * (0.45 + 0.55 * b)); L.show(lGone, A * ss(0.9, 1, b));
    L.show(lBar, A); L.show(lBarV, A); L.show(lMem, A); L.show(lMemV, A); L.show(lMm, A); L.show(lMmV, A);
    if (cur) {
      hud.readout('nodes', 'AD graph nodes (physics part)', `${os.adNodes.toLocaleString('en')}  (${os.adNodes ? '' : 'none'}${os.totalNodes} total)`.replace('  ()', ''), 'model estimate');
      hud.readout('mm', 'Matmuls per step', `${Math.round(mm)}   (${os.matmulsBefore} → ${os.matmulsAfter})`, 'model estimate');
      hud.readout('mem', 'AD-graph memory', fb(os.memBytes), 'model estimate');
      hud.readout('t', 'Step time', `${ft(os.total)}  (x${(os.totalBefore / os.total).toFixed(1)} vs before)`, 'model estimate');
    }
    void tmp;
  }
  function frame() {
    const mob = innerWidth <= 640, box = mob ? [-5.6, 5.6, -8.6, 9.0, -0.6, 0.6] : [-10.8, 10.2, -6.9, 4.6, -0.6, 0.6];
    const w = box.map((v, i) => (i < 2 ? v + Z[0] : i < 4 ? v + Z[1] : v + Z[2]));
    return frameBox(THREE, camera, w, { az: 0, el: mob ? 2 : 4, reserve: { top: mob ? 52 : 44 } });
  }
  const api = { get blend() { return S.blend; }, setBlend, get state() { return S.st; }, get before() { return before; }, get after() { return after; }, get auto() { return S.auto.on; },
    stats() { return { before: { nodes: before.stats.nodes, collapse: before.collapse, visible: before.group.visible }, after: { nodes: after.stats.nodes } }; } };
  return { group, update, frame, api, enter() { S.key = ''; S.placed = false; if (ctx.state.reducedMotion) { setBlend(1); S.auto.on = false; } else { setBlend(0); S.auto.on = true; S.auto.t = 0; } ctl.play.textContent = '❚❚ Pause'; },
    dispose() { offBlend?.(); ctl.dispose(); restore(); before.dispose(); after.dispose(); boxes.dispose(); util.disposeObject(group); } };
}
