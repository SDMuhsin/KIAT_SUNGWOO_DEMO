// cards.js - 2D interlude slides ("cards"): clean SVG/DOM explainers shown over the dimmed, persistent 3D scene.
// Each card introduces ONE idea before the harder 3D stage that follows it. All drawings are ILLUSTRATIVE sketches (no measured numbers).
// A card never changes the 3D scene: the previous stage's objects stay in place behind the veil (see stage.js: a card keeps the previous stage "effective").
const C = { w: '#4f8fe8', a: '#3ab795', g: '#f08a24', o: '#9b6bd6', p: '#e0408a', s: '#2fd0e6', gold: '#f2c14e', mut: '#8793a6', txt: '#e8edf5', line: '#2a3550', fill: '#121a2b' };
const T = (x, y, s, o = {}) => `<text x="${x}" y="${y}" fill="${o.c ?? C.txt}" font-size="${o.s ?? 22}" text-anchor="${o.a ?? 'middle'}"${o.b ? ' font-weight="700"' : ''}${o.i ? ' font-style="italic"' : ''}>${s}</text>`;
const R = (x, y, w, h, c, o = {}) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${o.r ?? 10}" fill="${o.f ?? C.fill}" stroke="${c}" stroke-width="${o.sw ?? 2.5}"${o.d ? ` stroke-dasharray="${o.d}"` : ''}/>`;
const L = (x1, y1, x2, y2, c = C.mut, o = {}) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${c}" stroke-width="${o.sw ?? 2.5}"${o.d ? ` stroke-dasharray="${o.d}"` : ''}${o.arrow ? ' marker-end="url(#ah)"' : ''}/>`;
const D = (x, y, r, c) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${c}"/>`;
const DEFS = `<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="${C.mut}"/></marker></defs>`;
const svg = (body, h = 360) => `<svg viewBox="0 0 720 ${h}" role="img" font-family="system-ui,Segoe UI,Roboto,sans-serif" preserveAspectRatio="xMidYMid meet">${DEFS}${body}</svg>`;

/** 1. Roofline primer: bytes vs FLOPs. */
function roofline() {
  const x0 = 70, y0 = 300, x1 = 690, ridge = 400, top = 90;
  const body = [
    L(x0, y0, x1, y0, C.mut, { arrow: true }), L(x0, y0, x0, 50, C.mut, { arrow: true }),
    T(380, 345, 'arithmetic intensity = FLOPs done per byte moved', { c: C.mut, s: 20 }),
    `<text x="22" y="190" fill="${C.mut}" font-size="20" text-anchor="middle" transform="rotate(-90 22 190)">speed (FLOP/s)</text>`,
    `<path d="M${x0} ${y0} L${ridge} ${top} L${x1} ${top}" fill="none" stroke="${C.s}" stroke-width="5" stroke-linejoin="round"/>`,
    `<path d="M${x0} ${y0} L${ridge} ${top} L${ridge} ${y0}z" fill="${C.p}" opacity=".10"/><path d="M${ridge} ${top} L${x1} ${top} L${x1} ${y0} L${ridge} ${y0}z" fill="${C.s}" opacity=".10"/>`,
    L(ridge, top, ridge, y0, C.gold, { d: '6 6' }), T(ridge, 78, 'ridge point', { c: C.gold, s: 20 }),
    T(215, 130, 'MEMORY-BOUND', { c: C.p, b: 1, s: 21 }), T(215, 155, 'waits for bytes', { c: C.p, s: 19 }),
    T(560, 175, 'COMPUTE-BOUND', { c: C.s, b: 1, s: 21 }), T(560, 200, 'waits for arithmetic', { c: C.s, s: 19 }),
    D(150, 262, 9, C.a), T(150, 288, 'activation', { s: 18, c: C.a }), D(262, 212, 9, C.p), T(300, 240, 'derivative', { s: 18, c: C.p }), D(610, 90, 9, C.w), T(610, 128, 'big matmul', { s: 18, c: C.w }),
  ].join('');
  return { title: 'How fast can a GPU run an operation?', html: svg(body),
    points: ['Every operation both <b>moves bytes</b> and <b>does FLOPs</b>. Intensity is the ratio of the two.', 'Low intensity (left of the ridge): the GPU idles waiting for memory, so more FLOP/s will not help.', 'High intensity (right): the arithmetic units are the limit. <b>Next:</b> where do OUR blocks sit?'],
    note: 'Illustrative sketch, not measured.' };
}

/** 2. PINN residual primer. */
function pinn() {
  const body = [
    R(14, 40, 120, 80, C.gold), T(74, 78, '(x, y, z)', { s: 22 }), T(74, 106, 'a sampled point', { s: 16, c: C.mut }),
    L(136, 80, 186, 80, C.mut, { arrow: true }),
    R(190, 40, 130, 80, C.w), T(255, 78, 'network', { s: 22, b: 1, c: C.w }), T(255, 106, 'weights θ', { s: 16, c: C.mut }),
    L(322, 80, 372, 80, C.mut, { arrow: true }),
    R(376, 40, 150, 80, C.a), T(451, 76, 'û = (u, v, w, p)', { s: 21, c: C.a }), T(451, 106, 'predicted field', { s: 16, c: C.mut }),
    L(528, 80, 570, 80, C.mut, { arrow: true }),
    R(574, 40, 132, 80, C.p), T(640, 70, 'PDE', { s: 22, b: 1, c: C.p }), T(640, 100, 'residual r', { s: 19, c: C.p }),
    T(360, 175, 'continuity (mass conservation):   r  =  ∂u/∂x + ∂v/∂y + ∂w/∂z   should be  0', { s: 21 }),
    L(60, 215, 660, 215, C.line),
    T(180, 250, 'r large', { s: 21, c: C.p, b: 1 }), D(80, 300, 7, C.p), D(120, 290, 11, C.p), D(165, 305, 16, C.p), D(215, 288, 13, C.p), D(255, 300, 9, C.p), T(180, 340, 'equation violated', { s: 18, c: C.mut }),
    L(330, 270, 400, 270, C.mut, { arrow: true }), T(365, 255, 'training', { s: 17, c: C.mut }),
    T(540, 250, 'r ≈ 0', { s: 21, c: C.s, b: 1 }), D(470, 300, 3, C.s), D(510, 292, 4, C.s), D(550, 304, 3, C.s), D(590, 296, 4, C.s), D(625, 301, 3, C.s), T(540, 340, 'equation satisfied', { s: 18, c: C.mut }),
  ].join('');
  return { title: 'What is a PINN residual?', html: svg(body),
    points: ['Plug the network output into the PDE. What is left over is the <b>residual</b>: zero means the physics holds at that point.', 'Training minimises <b>loss = data error + λ · mean(residual²)</b>, at every sampled interior point.', '<b>New cost:</b> the residual needs derivatives of the network, which the next slides unpack.'],
    note: 'Illustrative sketch. The real physics is RANS k-epsilon.' };
}

/** 3. Why the physics gradient graph is bigger. */
function graph() {
  const chain = [], lat = [];
  for (let i = 0; i < 5; i++) { const x = 40 + i * 62; chain.push(D(x, 150, 11, C.g)); if (i) chain.push(L(x - 62, 150, x, 150, C.g)); }
  for (let c = 0; c < 5; c++) for (let r = 0; r < 4; r++) {
    const x = 410 + c * 62, y = 90 + r * 38; lat.push(D(x, y, 9, C.p));
    if (c) { lat.push(L(x - 62, y, x, y, C.p, { sw: 1.8 })); if (r) lat.push(L(x - 62, y - 38, x, y, C.p, { sw: 1.8 })); if (r < 3) lat.push(L(x - 62, y + 38, x, y, C.p, { sw: 1.8 })); }
  }
  const body = [
    T(155, 38, 'Weight gradient', { s: 24, b: 1, c: C.g }), T(155, 66, 'dL/dθ : one chain', { s: 19, c: C.mut }), chain.join(''),
    T(155, 205, 'loss ← output ← layers ← weights', { s: 17, c: C.mut }),
    T(500, 38, 'Physics-loss gradient', { s: 24, b: 1, c: C.p }), T(500, 66, 'differentiates a derivative', { s: 19, c: C.mut }), lat.join(''),
    T(500, 270, 'u, then u_x, u_xx ... each', { s: 17, c: C.mut }), T(500, 292, 'is its own branch of the graph', { s: 17, c: C.mut }),
    L(300, 40, 300, 330, C.line, { d: '4 6' }),
    T(155, 300, 'few nodes', { s: 22, c: C.g, b: 1 }), T(500, 330, 'many more nodes', { s: 22, c: C.p, b: 1 }),
  ].join('');
  return { title: 'Why is the physics gradient graph bigger?', html: svg(body),
    points: ['A normal weight gradient is <b>one backward sweep</b> through the layers.', 'The physics loss contains <b>derivatives of the network</b> (u_x, u_xx ...). To train on it we must differentiate those derivatives again: a second-order graph that branches at every layer.', '<b>Next:</b> side by side, with the node counts from the cost model.'],
    note: 'Illustrative sketch. Real counts (model estimate) follow on the next slide.' };
}

/** 4. v and J split. */
function split() {
  const body = [
    R(250, 20, 220, 60, C.p), T(360, 58, 'loss L', { s: 24, b: 1, c: C.p }),
    L(360, 82, 360, 118, C.mut, { arrow: true }),
    R(250, 120, 220, 60, C.a), T(360, 158, 'output û', { s: 24, b: 1, c: C.a }),
    L(360, 182, 360, 218, C.mut, { arrow: true }),
    R(250, 220, 220, 60, C.w), T(360, 258, 'weights θ', { s: 24, b: 1, c: C.w }),
    `<path d="M248 50 C 150 50 130 100 130 150" fill="none" stroke="${C.s}" stroke-width="3"/>`,
    R(14, 110, 230, 120, C.s, { sw: 3 }), T(129, 145, 'v = ∂L / ∂û', { s: 24, b: 1, c: C.s }), T(129, 175, 'PDE + fixed matrices', { s: 18, c: C.mut }), T(129, 200, 'closed form, derived ONCE', { s: 18, c: C.gold }),
    R(476, 110, 230, 120, C.w, { sw: 3 }), T(591, 145, 'J = ∂û / ∂θ', { s: 24, b: 1, c: C.w }), T(591, 175, 'network Jacobian', { s: 18, c: C.mut }), T(591, 200, 'one ordinary backprop', { s: 18, c: C.gold }),
    T(360, 330, 'dL/dθ  =  v · J', { s: 30, b: 1, c: C.txt }),
  ].join('');
  return { title: 'SAGE: split the gradient in two', html: svg(body),
    points: ['The chain rule splits <b>dL/dθ = v · J</b>.', '<b>v</b> only knows the PDE and fixed matrices, so it can be derived symbolically <b>once</b> and emitted as code. <b>J</b> is the usual network backprop.', 'No second-order autograd graph is needed. <b>Next:</b> how v is derived (trace, tape, reverse walk).'],
    note: 'Illustrative sketch of the idea.' };
}

/** 5. Before/after bars. */
function bars() {
  const seg = (x, w, c, t) => `<rect x="${x}" y="0" width="${w}" height="52" fill="${c}" opacity=".85"/>` + (t ? `<text x="${x + w / 2}" y="32" fill="#0b0f17" font-size="17" text-anchor="middle" font-weight="700">${t}</text>` : '');
  const row = (y, parts) => `<g transform="translate(40 ${y})">${parts}</g>`;
  const body = [
    T(40, 40, 'One training step, schematic (not to scale)', { s: 21, c: C.mut, a: 'start' }),
    T(40, 98, 'PyTorch autograd', { s: 22, b: 1, c: C.p, a: 'start' }),
    row(112, seg(0, 100, C.a, 'fwd') + seg(100, 170, C.g, 'backward') + seg(270, 70, C.o, 'Adam') + seg(340, 150, C.p, 'graph build') + seg(490, 150, '#ff8cbf', 'traverse')),
    T(40, 214, 'SAGE', { s: 22, b: 1, c: C.s, a: 'start' }),
    row(228, seg(0, 100, C.a, 'fwd') + seg(100, 120, C.g, 'backward') + seg(220, 70, C.o, 'Adam')),
    `<g fill="none" stroke="${C.gold}" stroke-width="2.5" stroke-dasharray="6 5"><rect x="380" y="224" width="260" height="60" rx="8"/></g>`,
    T(510, 310, 'graph build + traverse: gone', { s: 20, c: C.gold }),
    T(40, 345, '~24 matmuls  →  10 matmuls (depth 4); no AD-graph memory', { s: 20, c: C.txt, a: 'start' }),
  ].join('');
  return { title: 'Before and after, in one bar', html: svg(body),
    points: ['The overhead blocks (building and traversing the autograd graph) are not arithmetic: they are bookkeeping.', 'SAGE removes them, keeps the three useful phases, and needs fewer matmuls.', '<b>Next:</b> the same comparison in 3D, with a slider and live model estimates.'],
    note: 'Schematic bar lengths, illustrative. Real numbers (model estimate) appear on the next slide.' };
}

export const CARDS = { roofline, pinn, graph, split, bars };

/** createCards() -> { show(id), hide(), el }. Builds the veil + panel once (DOM only; no three.js). */
export function createCards() {
  const veil = document.createElement('div'); veil.id = 'card-veil'; veil.setAttribute('aria-hidden', 'true');
  const panel = document.createElement('section'); panel.id = 'card-panel'; panel.className = 'card'; panel.setAttribute('aria-live', 'polite'); panel.hidden = true;
  document.body.append(veil, panel);
  const root = document.documentElement;
  const sync = () => { const b = document.getElementById('hud-bottom'); if (b) root.style.setProperty('--bottom-h', Math.ceil(b.getBoundingClientRect().height) + 'px'); };
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(sync).observe(document.getElementById('hud-bottom')); addEventListener('resize', sync); sync();
  let cur = null;
  return {
    el: panel,
    get current() { return cur; },
    show(id) {
      const c = CARDS[id]?.(); if (!c) return this.hide();
      cur = id; sync();
      panel.innerHTML = `<h2>${c.title}</h2><div class="card-fig">${c.html}</div><ul>${c.points.map((p) => `<li>${p}</li>`).join('')}</ul><p class="card-note">${c.note}</p>`;
      panel.hidden = false; panel.scrollTop = 0; document.body.classList.add('card-open');
      requestAnimationFrame(() => panel.classList.add('on'));
    },
    hide() { cur = null; document.body.classList.remove('card-open'); panel.classList.remove('on'); setTimeout(() => { if (!cur) panel.hidden = true; }, 380); },
  };
}
