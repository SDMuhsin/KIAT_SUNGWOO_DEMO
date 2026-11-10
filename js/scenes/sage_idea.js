// sage_idea.js - stage 12 (sage-idea): the SAGE idea. Gradient split dL/dtheta = v . J, then the one-time symbolic pipeline:
// PDE forward fn -> TracedVar tape (40 entries) -> reverse walk (VJP rules) -> emitted function -> compiled + cached ONCE -> called every step.
// Layout / animation ILLUSTRATIVE; the 40-entry tape count, the VJP rules and the "static, spectral-matrix PDE" prerequisite are from SAGE_EXPLAINED.tex.
import * as SM from './sage_math.js';
import { ZONES, createBoxes, createLabels, useStyle, hexCss } from './sage_kit.js';
import { frameBox } from './physics_cam.js';
import { makeControls } from './physics_ui.js';

const Z = ZONES.idea;
const STATIONS = [
  { n: 'PDE forward fn', s: 'run on fake inputs' }, { n: 'TracedVar tape', s: '40 entries' }, { n: 'reverse walk', s: 'VJP rules' },
  { n: 'generated fn', s: 'matmuls only' }, { n: 'compile + cache', s: 'exec() once' },
];
const LAYS = {
  d: { stx: [-7.2, -3.6, 0, 3.6, 7.2], sw: 3.0, sh: 1.1, sy: 0.25, nameY: 1.25, subY: -0.75, sub: true, split: { y: 3.2, grad: -7.4, eq: -5.9, v: -3.4, vw: 2.7, dot: -1.4, j: 1.6, jw: 2.7, lbl: 3.95, no: [6.3, 3.75], no2: [6.3, 2.85], vsY: 2.4 },
    sz: { grad: 0.75, vj: 0.6, no: 0.46, st: 0.5, sub: 0.38, loop: 0.46, tape: 0.46, vs: 0.4 }, names: null, loopTxt: ['\u00fb = net(x)', 'v = gen(\u00fb)  (cached)', 'pred.backward(v)'],
    tape: { cols: 20, x0: -8.2, dx: 0.86, y0: -2.2, dy: 0.62, w: 0.74, h: 0.5, lbl: [7.0, -1.65] }, loop: { y: -4.6, x: [-3.2, 0.9, 5.0], w: [3.2, 3.7, 3.2], calls: [-3.2, -5.6], once: [7.2, -0.75] }, box: [-9.4, 9.4, -6.1, 4.9] },
  m: { stx: [-4.2, -2.1, 0, 2.1, 4.2], sw: 1.9, sh: 0.95, sy: 0.0, nameY: 0.95, subY: 0, sub: false, split: { y: 3.1, grad: -4.2, eq: -3.05, v: -1.5, vw: 2.1, dot: -0.1, j: 1.4, jw: 2.1, lbl: 3.75, no: [0.0, 1.85], no2: null, vsY: 0 },
    sz: { grad: 0.85, vj: 0.7, no: 0.6, st: 0.55, sub: 0.38, loop: 0.55, tape: 0.55, vs: 0.4 }, names: ['PDE fn', 'tape', 'reverse', 'code', 'cache'], loopTxt: ['\u00fb = net', 'v = gen(\u00fb)', 'bwd(v)'],
    tape: { cols: 10, x0: -4.5, dx: 1.0, y0: -1.75, dy: 0.55, w: 0.86, h: 0.42, lbl: [3.2, -0.95] }, loop: { y: -4.6, x: [-3.4, 0, 3.4], w: [2.9, 3.3, 2.9], calls: [0, -5.5], once: [4.1, -0.55] }, box: [-5.4, 5.4, -6.1, 4.8] },
};
const DIM = 0x2a3550, C = { sage: 0x2fd0e6, gold: 0xf2c14e, blue: 0x4a90e2, orange: 0xf08a24, green: 0x3ab795 };
const mixHex = (a, b, t) => { const r = (h, s) => (h >> s) & 255, m = (s) => Math.round(r(a, s) + (r(b, s) - r(a, s)) * t); return (m(16) << 16) | (m(8) << 8) | m(0); };

export function createIdea(ctx, shared) {
  const { THREE, util, camera, knobs, hud } = ctx, { smoothstep: ss, clamp } = util;
  const group = new THREE.Group(); group.name = 'sage-idea'; group.position.set(...Z); group.visible = false;
  const boxes = createBoxes(THREE, 80); group.add(boxes.mesh);
  const labels = createLabels(THREE, util, group); const L = labels;
  const restore = useStyle();
  const tape = SM.buildTape(), walk = SM.reverseWalk(tape), code = SM.genCode(tape);
  // ---- labels
  const lGrad = L.add('dL/dθ', '#ff9a3c', 0.75), lEq = L.add('=', '#e8edf5', 0.75), lDot = L.add('·', '#e8edf5', 0.9);
  const lV = L.add('v = ∂L/∂û', '#2fd0e6', 0.6), lVs = L.add('PDE + fixed matrices only', '#9ee9f4', 0.4);
  const lJ = L.add('J = ∂û/∂θ', '#4a90e2', 0.6), lJs = L.add('ordinary backprop', '#a9c8f5', 0.4);
  const lNo = L.add('v: static, no weights in it', '#f2c14e', 0.46), lNo2 = L.add('pred.backward(gradient=v)', '#cfe3ff', 0.4);
  const lSt = STATIONS.map((s) => L.add(s.n, '#e8edf5', 0.5)), lSs = STATIONS.map((s) => L.add(s.s, '#aab4c6', 0.38));
  const lTape = L.add('tape 0 / 40', '#cfe3ff', 0.46);
  const lLoop = ['û = net(x)', 'v = gen(û)  (cached)', 'pred.backward(v)'].map((t) => L.add(t, '#e8edf5', 0.46));
  const lCalls = L.add('generated 1×', '#f2c14e', 0.46), lOnce = L.add('ONCE', '#2fd0e6', 0.4);
  // ---- DOM
  const ctl = makeControls('sage-idea-controls', 'SAGE idea animation controls', '#2fd0e6'); ctl.note.textContent = 'one-time pipeline'; ctl.note.style.display = 'none';
  const panel = document.createElement('aside'); panel.id = 'sage-console'; panel.className = 'card sg-panel'; panel.setAttribute('aria-label', 'SAGE pipeline console');
  panel.style.display = 'none'; panel.style.padding = '8px 10px';
  panel.innerHTML = '<h2><span class="sg-pill" id="sgc-pill">split</span><span id="sgc-title"></span></h2><div class="sg-code" id="sgc-code" aria-live="off"></div><div class="sg-chips" id="sgc-chips" role="list" aria-label="VJP rules"></div><div class="sg-note" id="sgc-note"></div>';
  document.body.appendChild(panel);
  const $ = (s) => panel.querySelector(s), elPill = $('#sgc-pill'), elTitle = $('#sgc-title'), elCode = $('#sgc-code'), elChips = $('#sgc-chips'), elNote = $('#sgc-note');
  const chips = SM.RULES.map((r) => { const c = document.createElement('span'); c.className = 'sg-chip'; c.setAttribute('role', 'listitem'); c.style.setProperty('--c', hexCss(SM.OPCOL[r.op])); c.textContent = `${r.name} → ${r.rule}`; c.title = r.math; elChips.appendChild(c); return c; });
  elNote.innerHTML = '<b>Prerequisite:</b> the PDE derivatives must be fixed (spectral / DT-PINN style) matrices, not autograd, so the structure is static. Without that, this does not apply.';
  // ---- state
  let LY = LAYS.d;
  const S = { clock: 0, playing: true, mob: null, vis: 0, shown: false, key: '', st: null, last: -1, calls: -1, done: false, frames: 0, userScrub: false };
  const tmp = new THREE.Color();
  ctl.replay.textContent = '\u21bb Replay'; ctl.replay.onclick = () => { S.clock = 0; S.playing = true; ctl.play.textContent = '❚❚ Pause'; S.done = false; };
  ctl.play.onclick = () => { S.playing = !S.playing; ctl.play.textContent = S.playing ? '❚❚ Pause' : '▶ Play'; };
  ctl.scrub.oninput = () => { S.clock = ctl.scrub.value / 1000 * SM.TL_END; S.playing = false; ctl.play.textContent = '▶ Play'; };

  function layout(mob) {
    S.mob = mob; const Y = mob ? LAYS.m : LAYS.d; LY = Y; const z = Y.sz, sp = Y.split;
    const rs = (l, size, text) => { l.userData.opts.size = size; l.userData.text = null; util.setLabel(l, text); };
    rs(lGrad, z.grad, 'dL/d\u03b8'); rs(lV, z.vj, 'v = \u2202L/\u2202\u00fb'); rs(lJ, z.vj, 'J = \u2202\u00fb/\u2202\u03b8'); rs(lNo, z.no, 'v: static, no weights in it'); rs(lNo2, 0.4, 'pred.backward(gradient=v)');
    rs(lVs, z.vs, 'PDE + fixed matrices only'); rs(lJs, z.vs, 'ordinary backprop'); rs(lEq, z.grad, '='); rs(lDot, z.grad * 1.2, '\u00b7');
    STATIONS.forEach((s0, i) => { rs(lSt[i], z.st, Y.names ? Y.names[i] : s0.n); rs(lSs[i], z.sub, s0.s); });
    Y.loopTxt.forEach((t, i) => rs(lLoop[i], z.loop, t)); rs(lTape, z.tape, lTape.userData.text ?? 'tape 0 / 40'); S.calls = -1;
    lGrad.position.set(sp.grad, sp.y, 0); lEq.position.set(sp.eq, sp.y, 0); lV.position.set(sp.v, sp.lbl, 0.3); lVs.position.set(sp.v, sp.vsY, 0.6); lDot.position.set(sp.dot, sp.y, 0);
    lJ.position.set(sp.j, sp.lbl, 0.3); lJs.position.set(sp.j, sp.vsY, 0.6); lNo.position.set(sp.no[0], sp.no[1], 0); if (sp.no2) lNo2.position.set(sp.no2[0], sp.no2[1], 0);
    STATIONS.forEach((s0, i) => { lSt[i].position.set(Y.stx[i], Y.nameY, 0.5); lSs[i].position.set(Y.stx[i], Y.subY, 0.5); });
    lTape.position.set(Y.tape.lbl[0], Y.tape.lbl[1], 0.3);
    lLoop.forEach((l, i) => l.position.set(Y.loop.x[i], Y.loop.y, 0.55)); lCalls.position.set(Y.loop.calls[0], Y.loop.calls[1], 0.4); lOnce.position.set(Y.loop.once[0], Y.loop.once[1], 0.5);
    rs(lCalls, 0.52, 'generated 1\u00d7'); rs(lOnce, 0.4, 'ONCE'); S.calls = -1;
  }

  function consoleText(st) {
    const win = (arr, n) => arr.slice(Math.max(0, arr.length - n));
    if (st.phase === 'split') return { pill: 'split', title: 'two independent factors', color: '#2fd0e6',
      lines: ['<u># chain rule, split in two</u>', '<i>dL/dθ = v · J</i>', '', 'v = ∂L/∂û   depends ONLY on the PDE', '                 and the fixed matrices', 'J = ∂û/∂θ   the network Jacobian', '', 'pred.backward(gradient=v)'] };
    if (st.phase === 'trace') {
      const n = st.tracedCount, ls = tape.slice(0, n).map((e) => `<u>${String(e.i).padStart(2)}</u>  ${e.text.padEnd(22)}<u>[${e.op}]</u>`);
      return { pill: 'trace', title: `TracedVar records the PDE: ${n} / 40`, color: '#f2c14e', lines: win(ls, 8) };
    }
    if (st.phase === 'reverse') {
      const n = st.walked, ls = walk.slice(0, Math.max(1, n + 1)).map((r) => `<u>${String(r.i).padStart(2)}</u>  ${(r.op).padEnd(6)}<i>${r.rule.padEnd(17)}</i>${r.lines[0] ?? ''}`);
      return { pill: 'reverse', title: `walk the tape backwards: ${Math.min(40, n + 1)} / 40`, color: '#f08a24', lines: win(ls, 8) };
    }
    if (st.phase === 'emit') {
      const n = clamp(Math.floor(st.emit * code.length), 1, code.length), ls = code.slice(0, n).map((l) => (l.trim().startsWith('#') ? `<u>${l}</u>` : l));
      return { pill: 'emit', title: `emit source code: ${n} lines`, color: '#3ab795', lines: [ls[0], ...win(ls.slice(1), 7)] };
    }
    if (st.phase === 'compile') return { pill: 'compile', title: 'compile once, cache', color: '#9b6bd6', lines: ['fn = exec(source)        <u># ONCE</u>', 'cache[pde_id] = fn', '', '<i>generated 1×</i>', 'no TracedVar, no tape, no graph left', 'at training time.'] };
    return { pill: 'training', title: `called every step: ${st.calls.toLocaleString('en')}×`, color: '#2fd0e6',
      lines: ['<u># every training step</u>', 'u_hat = net(x)', 'v = generated_backward(u_hat, g)  <u># cached</u>', 'u_hat.backward(gradient=v)        <u># one clean backprop</u>', 'opt.step()', '', `<u>generated 1×, called ${st.calls.toLocaleString('en')}× (deck example: 30,000)</u>`] };
  }

  function placeDom(mob) {
    const ht = document.getElementById('hud-top')?.getBoundingClientRect(), hb = document.getElementById('hud-bottom')?.getBoundingClientRect();
    const st = panel.style;
    if (!mob) { st.top = '12px'; st.right = '12px'; st.left = 'auto'; st.bottom = 'auto'; st.width = '318px'; st.maxHeight = Math.max(200, (hb?.top ?? 560) - 24) + 'px'; }
    else { const top = (hb?.top ?? 640) - 8; st.left = '8px'; st.right = '8px'; st.width = 'auto'; st.top = 'auto'; st.bottom = (innerHeight - top) + 'px'; st.maxHeight = '248px'; }
    void ht;
  }

  const lastLines = { v: '' };
  function update(dt, state, cur) {
    const idx = shared.idx.idea, r = Math.min(util.revealOf(state, 'sage.split'), 1), here = state.stageIndex === idx;
    S.vis += ((state.stageIndex > idx ? 0 : 1) - S.vis) * (state.reducedMotion ? 1 : 1 - Math.exp(-dt * 4)); if (Math.abs(S.vis - (state.stageIndex > idx ? 0 : 1)) < 0.003) S.vis = state.stageIndex > idx ? 0 : 1;
    const G = r * S.vis; group.visible = G > 0.002;
    const showDom = here && r > 0.5; ctl.show(showDom); const domOn = showDom ? 'block' : 'none'; if (panel.style.display !== domOn) { panel.style.display = domOn; if (showDom) { S.mob = null; } }
    if (!group.visible) return;
    const mob = innerWidth <= 640; if (mob !== S.mob) { layout(mob); placeDom(mob); }
    if ((S.frames++ % 20) === 0 && showDom) placeDom(mob);
    if (here) {
      if (state.reducedMotion) { if (!S.done) { S.clock = SM.TL_END; S.done = true; } } else if (S.playing) S.clock += dt;

    }
    const st = SM.ideaState(S.clock, SM.TAPE_LEN, code.length); S.st = st;
    if (document.activeElement !== ctl.scrub) ctl.scrub.value = Math.round(clamp(S.clock / SM.TL_END) * 1000);
    // ---------------- 3D
    const A = G, B = boxes;
    const sp = st.split, sv = ss(0, 0.45, sp), sj = ss(0.25, 0.7, sp), sl = ss(0.5, 1, sp);
    // gradient split blocks
    { const q = LY.split; B.set(0, q.v, q.y, 0, q.vw * sv, 1.1 * sv, 0.8 * sv, C.sage); B.set(1, q.j, q.y, 0, q.jw * sj, 1.1 * sj, 0.8 * sj, C.blue); }
    L.show(lGrad, A * sv); L.show(lEq, A * sv); L.show(lDot, A * sj); L.show(lV, A * sv); L.show(lVs, S.mob ? 0 : A * sv * 0.95); L.show(lJ, A * sj); L.show(lJs, S.mob ? 0 : A * sj * 0.95); L.show(lNo, A * sl); L.show(lNo2, S.mob ? 0 : A * sl * 0.95);
    // stations: activation levels
    const act = [ss(2.4, 3.2, S.clock) * (1 - ss(0, 1, st.traced / 40 - 0.98) * 0.35), st.phase === 'split' ? 0 : 1, st.phase === 'reverse' || st.walked > 0 ? 1 : 0, st.emit > 0 ? 1 : 0, st.compile];
    const live = [st.phase === 'trace' ? 1 : 0, st.phase === 'trace' ? 1 : 0, st.phase === 'reverse' ? 1 : 0, st.phase === 'emit' ? 1 : 0, st.phase === 'compile' || st.phase === 'loop' ? 1 : 0];
    const stCol = [C.gold, C.gold, C.orange, C.green, C.sage];
    for (let i = 0; i < 5; i++) {
      const on = clamp(act[i]), pulse = live[i] ? 1 + 0.06 * Math.sin(state.time * 6) : 1, show = ss(0.0, 0.6, sp) * 0.5 + 0.5;
      B.set(2 + i, LY.stx[i], LY.sy, 0, LY.sw * pulse * show, LY.sh * pulse * show, 0.8 * show, mixHex(DIM, stCol[i], on * (live[i] ? 1 : 0.8)));
      L.show(lSt[i], A * show); L.show(lSs[i], LY.sub ? A * show * 0.9 : 0);
    }
    // arrows between stations + moving packet
    const prog = st.phase === 'split' ? 0 : st.phase === 'trace' ? st.traced / 40 : st.phase === 'reverse' ? 1 + st.reverse : st.phase === 'emit' ? 2 + st.emit : st.phase === 'compile' ? 3 + st.compile : 4;
    for (let i = 0; i < 4; i++) { const lit = clamp(prog - i); B.set(7 + i, (LY.stx[i] + LY.stx[i + 1]) / 2, LY.sy, 0, S.mob ? 0.3 : 0.55, 0.09, 0.09, mixHex(DIM, C.sage, ss(0, 1, lit))); }
    const pk = clamp(prog, 0, 4), pi = Math.min(3, Math.floor(pk)), pt = pk - pi;
    B.set(11, SM.lerp(LY.stx[pi], LY.stx[pi + 1], pt), LY.sy, 0.5, 0.28, 0.28, 0.28, 0xffffff);
    if (prog <= 0 || prog >= 4) B.hide(11);
    // tape cells: 20 x 2
    const cur2 = st.cursor;
    for (let i = 0; i < 40; i++) {
      const T = LY.tape, col = i % T.cols, row = (i / T.cols) | 0, x = T.x0 + col * T.dx, y = T.y0 - row * T.dy, e = tape[i];
      const grown = ss(i - 0.2, i + 1.2, st.traced), walked = st.phase === 'reverse' ? (i > cur2 ? 1 : 0) : st.phase === 'split' || st.phase === 'trace' ? 0 : 1;
      const hot = i === cur2 ? 1 : 0, last = st.phase === 'trace' && Math.floor(st.traced) === i ? 1 : 0;
      const base = mixHex(SM.OPCOL[e.op], C.orange, walked * 0.65);
      B.set(12 + i, x, y, 0, T.w * grown * (1 + 0.25 * (hot + last)), T.h * grown * (1 + 0.25 * (hot + last)), 0.3 * grown, hot || last ? 0xffffff : mixHex(DIM, base, 0.35 + 0.65 * grown));
    }
    const tl = st.phase === 'split' || st.phase === 'trace' ? `tape ${Math.min(40, st.tracedCount)} / 40` : 'tape 40 / 40';
    if (S.last !== tl) { S.last = tl; L.set(lTape, tl); }
    L.show(lTape, A * ss(0, 0.2, st.traced / 40 + (st.phase !== 'split' ? 0.2 : 0)));
    // training loop row
    const lp = st.compiled, loopA = ss(0.15, 0.7, lp) * A, ph = st.loop * 1.8, hi = st.phase === 'loop' ? Math.floor(ph) % 3 : -1;
    const lx = LY.loop.x, ly = LY.loop.y;
    for (let i = 0; i < 3; i++) {
      const w = LY.loop.w[i], hot = hi === i ? 1 : 0;
      B.set(52 + i, lx[i], ly, 0, w * ss(0.15, 0.7, lp), 0.82 * ss(0.15, 0.7, lp), 0.5, mixHex(DIM, i === 1 ? C.sage : C.blue, 0.45 + 0.55 * hot));
      L.show(lLoop[i], loopA);
    }
    for (let i = 0; i < 2; i++) B.set(55 + i, (lx[i] + lx[i + 1]) / 2 + 0.1, ly, 0, 0.5 * loopA / (A || 1), 0.08, 0.08, C.sage);
    // cached lock tag + calls counter
    const callsQ = Math.floor(st.calls / 1000) * 1000, ctext = st.phase === 'loop' ? `generated 1×, called ${(st.calls >= 30000 ? 30000 : callsQ).toLocaleString('en')}×` : 'generated 1×';
    if (S.calls !== ctext) { S.calls = ctext; L.set(lCalls, ctext); }
    L.show(lCalls, loopA); L.show(lOnce, A * ss(0.2, 0.8, st.compile)); B.hide(57);
    B.commit(); B.mat.opacity = G;
    // ---------------- DOM console (only touch when changed)
    if (showDom) {
      const ct = consoleText(st), key = ct.pill + '|' + ct.title + '|' + ct.lines.join('\n');
      if (key !== S.key) {
        S.key = key; elPill.textContent = ct.pill; elPill.style.color = ct.color; elTitle.textContent = ct.title; elCode.innerHTML = ct.lines.join('\n');
        const act2 = st.phase === 'trace' ? tape[Math.max(0, st.tracedCount - 1)].op : st.phase === 'reverse' ? tape[Math.max(0, st.cursor)].op : null;
        chips.forEach((c, k) => c.classList.toggle('on', !!act2 && SM.RULES[k].op === act2));
      }
    }
    hud.readout('tape', 'Tape entries', `${Math.min(40, st.tracedCount)} / 40`, 'illustrative');
    void cur; void dt; void lastLines;
  }

  function frame() {
    const mob = innerWidth <= 640; placeDom(mob);
    const cH = mob ? Math.min(248, panel.getBoundingClientRect().height || 230) : 0, hb = document.getElementById('hud-bottom')?.getBoundingClientRect();
    const consoleTop = (hb?.top ?? 640) - 8 - cH, reserveBottom = mob ? Math.max(0, innerHeight - 250 - consoleTop) : 0;
    const box = [...(mob ? LAYS.m : LAYS.d).box, -0.6, 0.6].map((v, i) => (i < 2 ? v + Z[0] : i < 4 ? v + Z[1] : v + Z[2]));
    return frameBox(THREE, camera, box, { az: 0, el: 4, reserve: { top: 44, bottom: reserveBottom } });
  }
  const api = { get clock() { return S.clock; }, setClock(t) { S.clock = t; S.playing = false; }, get state() { return S.st; }, get phase() { return S.st?.phase; }, tapeLength: SM.TAPE_LEN, ruleCount: SM.RULES.length, codeLines: code.length,
    get consoleText() { return elCode.textContent; } };
  return { group, update, frame, enter() { S.clock = 0; S.playing = true; S.done = false; S.key = ''; ctl.play.textContent = '❚❚ Pause'; }, api,
    dispose() { ctl.dispose(); panel.remove(); restore(); boxes.dispose(); util.disposeObject(group); } };
}
