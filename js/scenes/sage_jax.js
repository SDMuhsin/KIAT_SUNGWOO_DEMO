// sage_jax.js - stage 14 (sage-jax): the JAX/compiled step + the Sep-2026 memory-traffic cuts.
// Toggles: (a) sageCompile = whole step in ONE GPU program (per-kernel launch gaps vanish), (b) sageHalf = half-width (bf16) storage between layers,
// (c) sageSinglePass = single-pass derivative bookkeeping. Roofline from stage 7 reappears (own mini plot); the bandwidth-bound op slides/shrinks.
// Deck-measured series (12.7 / 9.9 / 7.7 ms) is labelled "measured, Sep 2026 deck" ONLY for those exact states; everything else is MODEL ESTIMATE.
import * as SM from './sage_math.js';
import { ZONES, createBoxes, createLabels, useStyle } from './sage_kit.js';
import { frameBox } from './physics_cam.js';

const Z = ZONES.jax, OPC = { netDeriv: 0xe0408a, mmBwd: 0xf08a24, act: 0x3ab795, opt: 0x9b6bd6 }, OPN = { netDeriv: 'network + derivatives', mmBwd: 'clean backward', act: 'activations', opt: 'optimizer' };
const KN = { netDeriv: 6, mmBwd: 4, act: 3, opt: 1 }, NK = 14, DIMC = 0x1b2438, AI0 = -1, AI1 = 3, P0 = 10, P1 = 14.5;
const lerp = (a, b, t) => a + (b - a) * t;

export function createJax(ctx, shared) {
  const { THREE, util, camera, knobs, hud, model } = ctx, { smoothstep: ss, clamp } = util;
  const group = new THREE.Group(); group.name = 'sage-jax'; group.position.set(...Z); group.visible = false;
  const B = createBoxes(THREE, 160, { emissive: 0.25 }); group.add(B.mesh);
  const L = createLabels(THREE, util, group), restore = useStyle();
  knobs.define({ id: 'sageCompile', label: 'Compile whole step (one GPU program)', type: 'toggle', value: false, group: 'SAGE', help: 'Whole training step compiled into one GPU program: removes per-kernel launch overhead (model estimate).' });
  const lT = L.add('one training step', '#e8edf5', 0.4), lOver = L.add('launch overhead', '#f2c14e', 0.42), lProg = L.add('ONE GPU PROGRAM', '#2fd0e6', 0.4), lTot = L.add('', '#f2c14e', 0.4);
  const lMeas = L.add('measured, Sep 2026 deck', '#9ee9f4', 0.44);
  const nm = SM.SEP_SERIES.map((s) => L.add(s.label, '#cfe3ff', 0.42)), nv = SM.SEP_SERIES.map((s) => L.add(`${s.ms} ms`, '#e8edf5', 0.44));
  const lYou = L.add('your toggles', '#f2c14e', 0.42), lYouV = L.add('', '#f2c14e', 0.44);
  const lTr = L.add('traffic 126 units', '#e8edf5', 0.44);
  const lN1 = L.add('75% of the step = network + derivatives (Sep profiling)', '#aab4c6', 0.4), lN2 = L.add('bandwidth-bound: bf16 cut it by 1/5, not 1/2', '#aab4c6', 0.4);
  const lAx = L.add('arithmetic intensity (FLOP/byte) →', '#8793a6', 0.42), lAy = L.add('FLOP/s →', '#8793a6', 0.42), lRid = L.add('ridge', '#e8edf5', 0.42), lMB = L.add('memory-bound', '#e0408a', 0.44), lCB = L.add('compute-bound', '#2fd0e6', 0.44);
  const lRoof = L.add('roofline (model estimate)', '#e8edf5', 0.4), lOp = Object.keys(OPC).map((k) => L.add(OPN[k], '#' + OPC[k].toString(16).padStart(6, '0'), 0.4)), lGhost = L.add('all off', '#8793a6', 0.38);
  // DOM chip bar for mobile (the knob panel is collapsed there)
  const bar = document.createElement('div'); bar.id = 'sage-jax-chips'; bar.className = 'card'; bar.setAttribute('role', 'group'); bar.setAttribute('aria-label', 'SAGE optimisations');
  Object.assign(bar.style, { display: 'none', top: '86px', left: '12px', right: '12px', padding: '6px 8px', gap: '6px', flexWrap: 'wrap', zIndex: 5 });
  const chip = (id, txt) => { const b = document.createElement('button'); b.type = 'button'; b.className = 'sg-chipbtn'; b.textContent = txt; b.setAttribute('aria-pressed', 'false'); b.onclick = () => knobs.set(id, !knobs.get(id)); bar.appendChild(b); return b; };
  const chips = { sageCompile: chip('sageCompile', 'compile step'), sageHalf: chip('sageHalf', 'bf16 storage'), sageSinglePass: chip('sageSinglePass', 'single pass') };
  document.body.appendChild(bar);
  const offs = Object.keys(chips).map((id) => knobs.on(id, (v) => chips[id].setAttribute('aria-pressed', String(!!v))));

  const S = { vis: 0, mob: null, key: '', js: null, lt: 0, txt: {}, c: 0, tgt: null, cur: null, init: false };
  const E = { mk: new Float32Array(8), ghost: new Float32Array(2), kw: new Float32Array(NK), gap: 1, ridge: [0, 0], lbw: 0, lpk: 0, bars: new Float32Array(4), hl: new Float32Array(3) };

  const geo = (mob) => mob
    ? { px: -4.2, pw: 8.4, py: 1.9, ph: 4.4, rx: -4.6, rw: 9.2, ry: -0.9, bx: -4.6, bw: 5.8, by: [-3.0, -4.1, -5.2, -6.5], tr: -7.5, n1: -8.3, n2: -8.9 }
    : { px: 1.9, pw: 8.0, py: -3.6, ph: 6.6, rx: -10.2, rw: 8.6, ry: 3.0, bx: -10.2, bw: 5.8, by: [0.5, -0.6, -1.7, -3.0], tr: -4.2, n1: -4.95, n2: -5.55 };
  const left = (sp, x, y, z = 0.5) => sp.position.set(x + sp.scale.x / 2, y, z);

  function layout(mob) {
    S.mob = mob; const k = mob ? 1.28 : 1;
    const rs = (sp, base, t) => { sp.userData.opts.size = base * k; sp.userData.text = null; util.setLabel(sp, t); };
    const keep = (sp, base) => rs(sp, base, sp.userData.text ?? ' ');
    for (const sp of L.all) keep(sp, sp.userData.size);
    S.key = '';
  }
  const px = (g, ai) => g.px + (clamp(Math.log10(Math.max(1e-9, ai)), AI0, AI1) - AI0) / (AI1 - AI0) * g.pw;
  const py = (g, perf) => g.py + (clamp(Math.log10(Math.max(1, perf)), P0, P1) - P0) / (P1 - P0) * g.ph;

  function update(dt, state, cur) {
    const idx = shared.idx.jax, r = util.revealOf(state, 'sage.jax'), here = state.stageIndex === idx;
    const tv = state.stageIndex > idx ? 0 : 1; S.vis += (tv - S.vis) * (state.reducedMotion ? 1 : 1 - Math.exp(-dt * 4)); if (Math.abs(S.vis - tv) < 0.003) S.vis = tv;
    const G = r * S.vis; group.visible = G > 0.002;
    const mob = innerWidth <= 640, showBar = here && r > 0.5 && mob; if (bar.style.display !== (showBar ? 'flex' : 'none')) { bar.style.display = showBar ? 'flex' : 'none'; if (showBar) { const ht = document.getElementById('hud-top'); if (ht) bar.style.top = Math.round(ht.getBoundingClientRect().bottom + 6) + 'px'; } }
    if (!group.visible) return;
    if (mob !== S.mob) layout(mob);
    const cfg = ctx.params(), tog = { sageCompile: !!knobs.get('sageCompile'), sageHalf: !!knobs.get('sageHalf'), sageSinglePass: !!knobs.get('sageSinglePass') };
    Object.keys(chips).forEach((id) => { const v = String(!!knobs.get(id)); if (chips[id].getAttribute('aria-pressed') !== v) chips[id].setAttribute('aria-pressed', v); });
    const js = SM.jaxState(cfg, tog), js0 = SM.jaxState(cfg, {}); S.js = js; const g = geo(mob);
    // ------------- targets -> eased values
    const k = state.reducedMotion || !S.init ? 1 : 1 - Math.exp(-dt * 7);
    const tc = tog.sageCompile ? 1 : 0; S.c += (tc - S.c) * (state.reducedMotion ? 1 : 1 - Math.exp(-dt * 4)); if (Math.abs(S.c - tc) < 0.003) S.c = tc;
    js.ops.forEach((o, i) => { E.mk[i * 2] += (px(g, o.intensity) - E.mk[i * 2]) * k; E.mk[i * 2 + 1] += (py(g, o.perf) - E.mk[i * 2 + 1]) * k; });
    { const o0 = js0.ops[0]; E.ghost[0] = px(g, o0.intensity); E.ghost[1] = py(g, o0.perf); }
    const lbw = Math.log10(cfg.bandwidthGBs * 1e9), lpk = Math.log10(cfg.peakTflops * 1e12);
    if (!S.init) { E.lbw = lbw; E.lpk = lpk; }
    E.lbw += (lbw - E.lbw) * k; E.lpk += (lpk - E.lpk) * k;
    // rail kernel widths (fractions of the all-off total)
    const tot0 = js0.totalAllOff, RW = g.rw; let ki = 0;
    js.ops.forEach((o) => { const n = KN[o.id], w = RW * (o.time / tot0) / n; for (let q = 0; q < n; q++) { E.kw[ki] += (w - E.kw[ki]) * k; ki++; } });
    const gapW = RW * (js0.overhead / tot0) / NK; E.gap = gapW * (1 - S.c);
    const tgtBars = [SM.SEP_SERIES[0].ms, SM.SEP_SERIES[1].ms, SM.SEP_SERIES[2].ms, js.series.ms];
    for (let i = 0; i < 4; i++) E.bars[i] += (tgtBars[i] / SM.SEP_SERIES[0].ms * g.bw - E.bars[i]) * k;
    S.init = true;
    // ------------- boxes
    let n = 0;
    // roofline backdrop: memory-bound side (left of ridge) and compute-bound side
    const ridgeX = g.px + ((E.lpk - E.lbw) - AI0) / (AI1 - AI0) * g.pw, rx = clamp(ridgeX, g.px, g.px + g.pw);
    B.set(n++, (g.px + rx) / 2, g.py + g.ph / 2, -0.35, Math.max(1e-3, rx - g.px), g.ph, 0.05, 0x2a1020);
    B.set(n++, (rx + g.px + g.pw) / 2, g.py + g.ph / 2, -0.35, Math.max(1e-3, g.px + g.pw - rx), g.ph, 0.05, 0x0f2a33);
    // roof curve as dots: slope part up to the ridge, then flat
    for (let i = 0; i < 44; i++) {
      const u = i / 43, x = g.px + u * g.pw, la = AI0 + u * (AI1 - AI0), lp = Math.min(E.lpk, la + E.lbw);
      B.set(n++, x, g.py + (clamp(lp, P0, P1) - P0) / (P1 - P0) * g.ph, 0, 0.1, 0.1, 0.1, 0xe8edf5);
    }
    // ridge marker (vertical dots)
    for (let i = 0; i < 8; i++) B.set(n++, rx, g.py + (i + 0.5) / 8 * g.ph, -0.1, 0.045, 0.2, 0.045, 0x8793a6);
    // ghost (all-off) marker + trail + current markers
    B.set(n++, E.ghost[0], E.ghost[1], 0.05, 0.3, 0.3, 0.1, 0x55607a);
    const mx = E.mk[0], my = E.mk[1];
    for (let i = 1; i < 7; i++) { const t = i / 7; B.set(n++, lerp(E.ghost[0], mx, t), lerp(E.ghost[1], my, t), 0.03, 0.07, 0.07, 0.07, 0xf2c14e); }
    const i2 = js.ops.map((o) => o.id);
    js.ops.forEach((o, i) => B.set(n++, E.mk[i * 2], E.mk[i * 2 + 1], 0.2, i === 0 ? 0.46 : 0.3, i === 0 ? 0.46 : 0.3, 0.3, o.bound === 'compute' ? 0x2fd0e6 : OPC[o.id]));
    // rail: 14 kernel blocks + gaps; compile frame
    const ry = g.ry, kw = E.kw; let x = g.rx; ki = 0;
    B.set(n++, g.rx + RW / 2, ry, -0.2, RW + 0.16, 1.15, 0.1, DIMC);
    const frameW = Math.max(0.01, (kw.reduce((a, v) => a + v, 0) + E.gap * (NK - 1))) + 0.2;
    B.set(n++, g.rx + frameW / 2 - 0.1, ry, -0.12, frameW, 1.15 * S.c + 1e-4, 0.08, 0x0e3a44);
    js.ops.forEach((o, oi) => { for (let q = 0; q < KN[o.id]; q++) { const w = kw[ki]; B.set(n++, x + w / 2, ry, 0, Math.max(w - 0.02 * (1 - S.c), 1e-4), 0.8, 0.45, o.bound === 'compute' ? 0x2fd0e6 : OPC[o.id]); x += w; if (ki < NK - 1) { B.set(n++, x + E.gap / 2, ry, 0, E.gap, 0.8, 0.45, 0xf2c14e); x += E.gap; } else n++; ki++; } });
    // measured series bars (3) + your toggles bar
    const act = js.series.idx;
    for (let i = 0; i < 4; i++) {
      const w = E.bars[i], y = g.by[i], hot = i === 3 ? 0 : (act === i ? 1 : 0);
      B.set(n++, g.bx + w / 2, y, 0, w, 0.42, 0.4, i === 3 ? (js.series.measured ? 0x5aa0ff : 0xf2c14e) : hot ? 0x2fd0e6 : 0x30405f);
      if (i < 3) B.set(n++, g.bx + g.bw + 0.2, y, 0, 0.14, 0.14 + 0.2 * hot, 0.14, hot ? 0x2fd0e6 : 0x1b2438); else n++;
    }
    for (; n < 160; n++) B.hide(n);
    B.used = 160; B.commit(); B.mat.opacity = G;
    // ------------- labels
    S.lt += dt; const ft = model.fmtTime;
    if (S.lt > 0.1) {
      S.lt = 0; const set = (sp, key, t) => { if (S.txt[key] !== t) { S.txt[key] = t; L.set(sp, t); } };
      set(lTot, 'tot', `${ft(js.total)}  (all off ${ft(js0.total)})`);
      set(lYouV, 'you', `${js.series.ms.toFixed(1)} ms  ${js.series.measured ? '(measured)' : '(model estimate)'}`);
      set(lYou, 'yl', js.series.measured ? 'your toggles = a deck figure' : 'your toggles: model estimate');
      set(lTr, 'tr', `traffic ${Math.round(js.series.units)} units (126 → 31)${js.series.trafficMeasured ? '' : ' illustrative split'}`);
    }
    const A = G;
    L.show(lT, A); L.show(lOver, A * clamp(1 - S.c + 0.15)); L.show(lProg, A * S.c); L.show(lTot, 0); L.show(lMeas, A);
    L.show(lAx, A); L.show(lAy, A); L.show(lRid, A); L.show(lMB, A * 0.9); L.show(lCB, A * 0.9); L.show(lRoof, A); L.show(lGhost, S.mob || Math.hypot(E.mk[0] - E.ghost[0], E.mk[1] - E.ghost[1]) < 1.1 ? 0 : A * 0.8);
    for (let i = 0; i < 3; i++) { L.show(nm[i], A); L.show(nv[i], A); }
    L.show(lYou, A); L.show(lYouV, A); L.show(lTr, A); L.show(lN1, A); L.show(lN2, A);
    lOp.forEach((l, i) => L.show(l, A * (i === 0 || !mob ? 1 : 0.85)));
    // positions
    left(lT, g.rx, ry + 1.05); lOver.position.set(g.rx + RW - lOver.scale.x / 2, ry + 1.05, 0.5); lProg.position.set(g.rx + RW - lProg.scale.x / 2, ry - 0.95, 0.6); left(lTot, g.rx, ry - 1.45);
    left(lMeas, g.bx, g.by[0] + (mob ? 1.45 : 1.0));
    for (let i = 0; i < 3; i++) { left(nm[i], g.bx, g.by[i] + 0.5, 0.8); left(nv[i], g.bx + E.bars[i] + 0.4, g.by[i]); }
    left(lYou, g.bx, g.by[3] + 0.5, 0.8); left(lYouV, g.bx + E.bars[3] + 0.4, g.by[3]);
    left(lTr, g.bx, g.tr); left(lN1, g.bx, g.n1); left(lN2, g.bx, g.n2);
    lAx.position.set(g.px + g.pw / 2, g.py - 0.5, 0.5); lAy.position.set(g.px - 0.15 + (mob ? 0.4 : 0), g.py + g.ph + 0.5, 0.5);
    lRid.position.set(rx, g.py + g.ph + 0.4, 0.5); lRoof.position.set(g.px + g.pw / 2, g.py + g.ph + (mob ? 1.0 : 1.0), 0.5);
    lMB.position.set(mob ? Math.max(g.px + lMB.scale.x / 2 + 0.1, rx - lMB.scale.x / 2 - 0.15) : g.px + Math.max(lMB.scale.x / 2 + 0.1, (rx - g.px) / 2), g.py + 0.45, 0.5); lCB.position.set(Math.min(g.px + g.pw - lCB.scale.x / 2 - 0.1, (rx + g.px + g.pw) / 2 + 0.5), g.py + 0.45, 0.5);
    lGhost.position.set(E.ghost[0] - 0.1, E.ghost[1] - 0.45, 0.5);
    lOp.forEach((l, i) => l.position.set(Math.min(g.px + g.pw - l.scale.x / 2, Math.max(g.px + l.scale.x / 2, E.mk[i * 2])), E.mk[i * 2 + 1] + (i === 0 ? 0.55 : (i % 2 ? (mob ? -0.9 : -0.45) : (mob ? 0.95 : 0.45))), 0.6));
    void i2; void cur;
    if (cur) {
      const se = js.series;
      hud.readout('stepm', 'Step time (deck)', se.measured ? `${se.ms.toFixed(1)} ms` : 'n/a for this combination', se.measured ? 'measured, Sep 2026 deck' : 'no deck figure');
      hud.readout('stepmodel', 'Step (model)', `${ft(js.total)}  x${js.speedupVsAllOff.toFixed(2)} vs all off`, 'model estimate');
      hud.readout('traffic', 'Memory traffic', `${Math.round(se.units)} units` + (se.trafficMeasured ? '' : ' (split illustrative)'), se.trafficMeasured ? 'deck endpoint' : 'model estimate');
      hud.readout('nd', 'Network + deriv. op', `${js.bound}-bound, ${(js.netDerivShare * 100).toFixed(0)}% of modelled step (deck: 75%)`, 'model estimate');
      hud.readout('launch', 'Kernel launches / step', `${js.launches}${tog.sageCompile ? ' (one program)' : ''}`, 'model estimate');
    }
  }
  function frame() {
    const mob = innerWidth <= 640, box = mob ? [-5.6, 5.6, -9.3, 7.6, -0.6, 0.6] : [-10.8, 10.4, -6.4, 4.8, -0.6, 0.6];
    return frameBox(THREE, camera, box.map((v, i) => (i < 2 ? v + Z[0] : i < 4 ? v + Z[1] : v + Z[2])), { az: 0, el: mob ? 2 : 4, reserve: { top: mob ? 44 : 0 } });
  }
  const api = { get state() { return S.js; }, get series() { return S.js?.series; }, get markers() { return Array.from(E.mk); }, get compileEase() { return S.c; } };
  return { group, update, frame, api, enter() { S.init = false; S.key = ''; },
    dispose() { offs.forEach((f) => f?.()); bar.remove(); restore(); B.dispose(); util.disposeObject(group); } };
}
