// stage.js - STAGES list + stage controller (progressive disclosure engine). See llmdocs/CONTRACT_NOTES.md.
import { easeInOutCubic } from './util.js';

/**
 * Ordered stage list (order and content fixed by SPEC). Fields:
 *  id, title, caption          - HUD text (a scene handler may override title/caption)
 *  module                      - scene module that owns this stage's handler ('geometry'|'network'|'memory'|'physics'|'sage')
 *  parts                       - NEW part ids revealed at this stage (cumulative: stays revealed in later stages)
 *  hide                        - optional part ids to fade OUT at this stage (rarely needed)
 *  camera {pos,target}         - default glide target (a handler may supply `camera` to override)
 *  knobs                       - knob ids shown in the panel
 *  badges                      - subset of ['illustrative','model estimate','projected']
 * Part ids are 'module.part' strings; state.reveal[partId] is eased 0..1.
 */
export const STAGES = [
  { id: 'geometry', title: '1. The geometry', short: 'The geometry', why: '', module: 'geometry', parts: ['geometry.duct', 'geometry.fieldslot'], knobs: [], badges: ['illustrative'],
    caption: 'A battery-pack-like cooling duct with inlet and outlet. Beside it, an empty slot for the field the network will predict.', camera: { pos: [0, 5, 16], target: [0, 0, 0] } },
  { id: 'sampling', title: '2. Sampling points', short: 'Sampling', why: 'the network needs points (x, y, z) to evaluate, not a mesh', module: 'geometry', parts: ['sampling.interior', 'sampling.boundary'], knobs: ['batch'], badges: ['illustrative'],
    caption: 'Interior collocation points and boundary points are sampled from the geometry. Each is an (x, y, z) input.', camera: { pos: [-2, 5, 14], target: [0, 0, 0] } },
  { id: 'network', title: '3. The network', short: 'The network', why: 'those points need a model that maps (x, y, z) to the flow field', module: 'network', parts: ['network.layers', 'network.edges'], knobs: ['width', 'depth', 'activation'], badges: ['illustrative'],
    caption: 'An MLP drawn in the same space: planes of neurons, edges with thickness proportional to |weight|.', camera: { pos: [4, 5, 18], target: [3, 0, 0] } },
  { id: 'forward', title: '4. Forward pass', short: 'Forward pass', why: 'the network only matters once the points flow through it and a field comes out', module: 'network', parts: ['forward.pulse', 'forward.field'], knobs: ['batch', 'width', 'depth', 'activation'], badges: ['illustrative'],
    caption: 'Points travel through the layers; the output field (u, v, w, p) resolves on the geometry.', camera: { pos: [6, 4, 17], target: [4, 0, 0] } },
  { id: 'memory-fwd', title: '5. Memory: weights and activations', short: 'Memory', why: 'every layer we just ran leaves weights and activations in GPU memory', module: 'memory', parts: ['memory.weights', 'memory.activations'], knobs: ['batch', 'width', 'depth', 'precision'], badges: ['model estimate'],
    caption: 'Memory blocks appear: WEIGHTS and ACTIVATIONS stacked per layer, volume proportional to bytes.', camera: { pos: [4, 6, 18], target: [3, -1, 0] } },
  { id: 'backward', title: '6. Backward pass', short: 'Backward pass', why: 'training also needs gradients, and they consume those saved activations', module: 'memory', parts: ['backward.flow', 'memory.gradients', 'memory.optimizer'], knobs: ['batch', 'width', 'depth', 'precision'], badges: ['model estimate'],
    caption: 'Autograd walks output to input, consuming activations; GRADIENTS appear, and the Adam OPTIMIZER STATE (m, v) joins them.', camera: { pos: [5, 6, 19], target: [3, -1, 0] } },
  { id: 'roofline-primer', kind: 'card', card: 'roofline', title: '7. Primer: bytes vs FLOPs', short: 'Primer: roofline', why: 'to judge speed we need one idea first, bytes moved versus FLOPs done', module: null, parts: [], knobs: [], badges: ['illustrative'],
    caption: 'One idea before the next 3D view: an operation is limited either by the bytes it moves or by the arithmetic it does.' },
  { id: 'bounds', title: '8. Compute- vs memory-bound', short: 'Compute vs memory', why: 'now we can ask where the memory blocks you just saw sit on that roofline', module: 'memory', parts: ['bounds.overlay'], knobs: ['batch', 'width', 'depth', 'precision', 'bandwidth', 'peakTflops'], badges: ['model estimate'],
    caption: 'Roofline view, drawn under the memory blocks: each operation coloured by arithmetic intensity against the ridge point; move the knobs and watch bounds flip.', camera: { pos: [3, 8, 20], target: [3, -1, 0] } },
  { id: 'pinn-primer', kind: 'card', card: 'pinn', title: '9. Primer: the PDE residual', short: 'Primer: residual', why: 'a physics-informed network adds a new loss term, the PDE residual', module: null, parts: [], knobs: [], badges: ['illustrative'],
    caption: 'One idea before the 3D view: plug the predicted field into the PDE and measure what is left over.' },
  { id: 'pinn', title: '10. Physics constraints (PINN)', short: 'Physics constraints', why: 'the residual is evaluated at every interior point of the same output field', module: 'physics', parts: ['pinn.constraints', 'pinn.residualfield'], knobs: ['nResiduals', 'lambda'], badges: ['illustrative'],
    caption: 'The PDE residual (continuity, momentum, k-epsilon) becomes extra loss: the field is violated, then satisfied.', camera: { pos: [-3, 5, 16], target: [0, 0, 0] } },
  { id: 'derivs', title: '11. Derivatives w.r.t. inputs', short: 'Input derivatives', why: 'the residual needs derivatives of the output with respect to the inputs', module: 'physics', parts: ['derivs.tower'], knobs: ['pdeOrder', 'nResiduals', 'depth'], badges: ['illustrative', 'model estimate'],
    caption: 'The residual needs u_x, u_xx, ... of the network with respect to its INPUTS: a derivative tower multiplying memory and compute.', camera: { pos: [-1, 6, 15], target: [1, 1, 0] } },
  { id: 'graph-primer', kind: 'card', card: 'graph', title: '12. Primer: why the graph grows', short: 'Primer: gradient graph', why: 'training on those derivatives means differentiating a derivative', module: null, parts: [], knobs: [], badges: ['illustrative'],
    caption: 'One idea before the comparison: a gradient through derivatives is a much bigger graph than a gradient through plain layers.' },
  { id: 'phys-grad', title: '13. Weight gradient vs physics gradient', short: 'Weight vs physics gradient', why: 'we can now compare the two gradient graphs side by side', module: 'physics', parts: ['physgrad.weightpath', 'physgrad.physpath'], knobs: ['pdeOrder', 'nResiduals', 'depth'], badges: ['illustrative', 'model estimate'],
    caption: 'Left: the weight gradient, one simple chain. Right: the physics-loss gradient differentiates through derivatives of the network, a branching second-order graph.', camera: { pos: [0, 4, 15], target: [0, 0, 0] } },
  { id: 'autograd-cost', title: '14. Autograd cost of physics', short: 'Autograd cost', why: 'that bigger graph costs real time and memory in every training step', module: 'physics', parts: ['autograd.graph', 'autograd.cost'], knobs: ['pdeOrder', 'nResiduals', 'depth', 'batch'], badges: ['model estimate'],
    caption: 'The full autograd graph for the physics loss: nodes, stored intermediates, traversals; memory and time inflate versus plain supervised training.', camera: { pos: [2, 6, 17], target: [1, 0, 0] } },
  { id: 'split-primer', kind: 'card', card: 'split', title: '15. Primer: split the gradient', short: 'Primer: v and J', why: 'SAGE avoids that graph by splitting the gradient in two', module: null, parts: [], knobs: [], badges: ['illustrative'],
    caption: 'One idea before SAGE: the physics gradient factors into a PDE-only part v and the ordinary network Jacobian J.' },
  { id: 'sage-idea', title: '16. SAGE: derive v once', short: 'Derive v once', why: 'v can be derived once, symbolically, and emitted as code', module: 'sage', parts: ['sage.split', 'sage.tape'], knobs: [], badges: ['illustrative'],
    caption: 'v = dL/d(u-hat) is PDE-only and closed form, derived symbolically ONCE (trace, tape, reverse-walk VJP, emit code); then one clean backprop through the network Jacobian.', camera: { pos: [0, 4, 15], target: [0, 0, 0] } },
  { id: 'bars-primer', kind: 'card', card: 'bars', title: '17. Primer: before and after', short: 'Primer: before/after', why: 'next we measure what removing the graph saves', module: null, parts: [], knobs: [], badges: ['illustrative'],
    caption: 'One idea before the 3D comparison: the overhead blocks of a step are bookkeeping, and SAGE removes them.' },
  { id: 'sage-overhead', title: '18. SAGE: overhead removed', short: 'Overhead removed', why: 'the autograd graph disappears from the step', module: 'sage', parts: ['sage.before_after'], knobs: ['sageBlend', 'batch', 'width', 'depth', 'pdeOrder', 'nResiduals', 'bandwidth'], badges: ['model estimate'],
    caption: 'Before and after: the autograd graph vanishes, 24 matmuls become 10, no AD-graph memory.', camera: { pos: [0, 5, 17], target: [0, 0, 0] } },
  { id: 'sage-jax', title: '19. SAGE: one compiled program', short: 'Compiled program', why: 'what remains is bandwidth-bound, so compile it and cut memory traffic', module: 'sage', parts: ['sage.jax'], knobs: ['sageCompile', 'sageHalf', 'sageSinglePass', 'bandwidth', 'peakTflops', 'precision'], badges: ['model estimate', 'illustrative'],
    caption: 'A single compiled GPU program plus half-width storage and single-pass bookkeeping: traffic 126 to 31 units, 12.7 to 7.7 ms/step (Sep profiling).', camera: { pos: [2, 4, 15], target: [1, 0, 0] } },
  { id: 'sage-results', title: '20. Results, honestly', short: 'Results', why: 'it is time to put honest end-to-end numbers on all of this', module: 'sage', parts: ['sage.results'], knobs: ['showProjected'], badges: ['projected'],
    caption: '50x per step vs PyTorch as UofW runs it; whole fine-tune run 10-16x; bedrock ~40x is PROJECTED, not measured.', camera: { pos: [0, 3, 14], target: [0, 0, 0] } },
];

/** Cards are 2D interlude slides: no 3D handler, no parts; the previous 3D stage stays "effective" (state.stageIndex) behind a veil. */
export const isCard = (st) => st?.kind === 'card';
/** Index of the 3D stage that is live while slide i is shown (cards inherit the previous non-card slide). */
export const effIndex = (i) => { while (i > 0 && isCard(STAGES[i])) i--; return i; };
/** One-line bridge between slides. dir = +1 forward, -1 backward. */
export function bridgeText(i, dir = 1) {
  const st = STAGES[i], p = STAGES[i - 1], n = STAGES[i + 1];
  if (dir < 0 && n) return `Back from "${n.short}" to "${st.short}".`;
  if (!p) return 'Start here: the design we want to simulate.';
  return `Last: ${p.short}. Now: ${st.short}, because ${st.why}.`;
}

/** Cumulative set of revealed part ids at stage index i. */
export function partsAt(i) {
  const s = new Set();
  for (let k = 0; k <= i; k++) { STAGES[k].parts.forEach((p) => s.add(p)); (STAGES[k].hide ?? []).forEach((p) => s.delete(p)); }
  return s;
}
export const ALL_PARTS = [...new Set(STAGES.flatMap((s) => s.parts))];
export const stageIndexOf = (id) => STAGES.findIndex((s) => s.id === id);

/**
 * createStageController({ctx, modules, hudEls, cards, hooks}) - owns state.reveal, camera glide, HUD, hash, keys, swipe.
 * modules: { [moduleId]: {group, stages:{[stageId]:handler}, dispose} }.  Returns {goTo,next,prev,update,index,dispose}.
 * Continuity rules: (1) state.stageIndex/stageId is the EFFECTIVE 3D stage (a card keeps the previous 3D stage live under a veil);
 * state.slideIndex is the slide shown. (2) On a long camera move the stage "commit" (reveal targets + enter()) is deferred until the
 * glide is LAG_T done, so the previous scene stays visible while the camera leaves it and the next one fades in as the camera arrives.
 * (3) Glide duration grows with distance and the path bows outward (pull-back arc) so both scenes stay in view mid-flight.
 */
const LAG_T = 0.62, LAG_DIST = 9;
export function createStageController({ ctx, modules, hudEls, cards, hooks }) {
  const { state, camera, controls, knobs, hud } = ctx;
  state.reveal = Object.fromEntries(ALL_PARTS.map((p) => [p, 0]));
  state.stageIndex = -1; state.stageId = null; state.prevIndex = -1; state.slideIndex = -1; state.time = 0; state.dt = 0;
  let force = 0, glide = null, active = null, writingHash = false, slide = -1, pending = null, from = -1;
  const handlerOf = (st) => (st.module ? modules[st.module]?.stages?.[st.id] : null);
  const targets = () => partsAt(state.stageIndex);

  function cameraFor(st) { return handlerOf(st)?.camera ?? st.camera; }
  /** Starts the camera glide; returns the travelled distance (world units). */
  function startGlide(cam, instant) {
    const p1 = camera.position.clone().set(...cam.pos), t1 = controls.target.clone().set(...cam.target);
    const dist = camera.position.distanceTo(p1) + controls.target.distanceTo(t1);
    if (instant || state.reducedMotion) { camera.position.copy(p1); controls.target.copy(t1); controls.update(); glide = null; return dist; }
    glide = { t: 0, dur: Math.min(3.2, Math.max(1.4, 1.2 + 0.04 * dist)), arc: Math.min(12, Math.max(0, 0.2 * dist - 2)), p0: camera.position.clone(), t0: controls.target.clone(), p1, t1 };
    return dist;
  }
  let moved = false, rzT = 0;
  controls.addEventListener('start', () => { glide = null; moved = true; });
  // re-frame on window resize (only if the user has not orbited away since entering this stage)
  const onResize = () => { clearTimeout(rzT); rzT = setTimeout(() => { if (!moved && state.stageIndex >= 0) startGlide(cameraFor(STAGES[state.stageIndex]), true); }, 120); };
  addEventListener('resize', onResize);

  /** Make stage e the live 3D stage: exit the old handler, set reveal targets, call enter(). */
  function commit(e, instant) {
    pending = null; if (e === state.stageIndex) return;
    const st = STAGES[e];
    if (active) { try { active.exit?.(); } catch (err) { console.error('stage exit', err); } }
    state.prevIndex = state.stageIndex; state.stageIndex = e; state.stageId = st.id;
    hud.clearReadouts(); force = 2;
    if (instant || state.reducedMotion) { const tgt = targets(); for (const p of ALL_PARTS) state.reveal[p] = tgt.has(p) ? 1 : 0; }
    moved = false; active = handlerOf(st) ?? null;
    try { active?.enter?.(instant ? 1 : 0); } catch (err) { console.error('stage enter', err); }
  }

  function goTo(target, { instant = false, fromHash = false } = {}) {
    const i = typeof target === 'number' ? target : stageIndexOf(target);
    if (i < 0 || i >= STAGES.length) return false;
    if (i === slide) return true;
    if (pending) commit(pending.e, false);            // a previous deferred move: land it first
    const dir = i > slide ? 1 : -1, st = STAGES[i], e = effIndex(i);
    if (slide >= 0 && !instant) hooks?.leave?.();
    from = slide; slide = i; state.slideIndex = i;
    knobs.setVisible(st.knobs);
    renderHud(st, dir);
    if (isCard(st)) cards?.show(st.card); else cards?.hide();
    if (e !== state.stageIndex) {                      // 3D stage changes (a card inside the same 3D stage does not move anything)
      const dist = startGlide(cameraFor(STAGES[e]), instant);
      if (instant || state.reducedMotion || dist < LAG_DIST) commit(e, instant); else pending = { e };
    }
    if (!fromHash) { writingHash = true; history.replaceState(null, '', '#stage=' + st.id); setTimeout(() => (writingHash = false), 0); }
    return true;
  }
  const next = () => goTo(Math.min(STAGES.length - 1, slide + 1));
  const prev = () => goTo(Math.max(0, slide - 1));

  function renderHud(st, dir) {
    const h = isCard(st) ? null : handlerOf(st);
    hudEls.title.textContent = h?.title ?? st.title; hudEls.caption.textContent = h?.caption ?? st.caption;
    if (hudEls.bridge) hudEls.bridge.textContent = bridgeText(slide, dir);
    hudEls.count.textContent = `${slide + 1} / ${STAGES.length}`;
    hudEls.bar.style.width = ((slide + 1) / STAGES.length * 100) + '%';
    hudEls.prev.disabled = slide === 0; hudEls.next.disabled = slide === STAGES.length - 1;
    hudEls.dots.querySelectorAll('button').forEach((b, k) => { b.setAttribute('aria-current', k === slide ? 'step' : 'false'); b.classList.toggle('done', k < slide); });
    hud.setBadges(st.badges ?? []);
    document.title = `SAGE demo - ${st.title}`;
  }

  // dots (cards are drawn as small squares)
  STAGES.forEach((st, k) => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'dot' + (isCard(st) ? ' card-dot' : ''); b.title = st.title; b.setAttribute('aria-label', 'Go to ' + st.title);
    b.onclick = () => goTo(k); hudEls.dots.appendChild(b);
  });
  hudEls.prev.onclick = prev; hudEls.next.onclick = next;

  const onKey = (e) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const tag = e.target.tagName; if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (e.key === ' ' && tag === 'BUTTON') return;
    if (e.key === 'ArrowRight' || (e.key === ' ' && !e.shiftKey) || e.key === 'PageDown') { next(); e.preventDefault(); }
    else if (e.key === 'ArrowLeft' || (e.key === ' ' && e.shiftKey) || e.key === 'PageUp') { prev(); e.preventDefault(); }
    else if (e.key === 'Home') { goTo(0); e.preventDefault(); }
    else if (e.key === 'End') { goTo(STAGES.length - 1); e.preventDefault(); }
  };
  addEventListener('keydown', onKey);
  // swipe on the HUD bottom bar (canvas swipes belong to OrbitControls)
  let sx = null;
  hudEls.swipe.addEventListener('touchstart', (e) => { sx = e.touches[0].clientX; }, { passive: true });
  hudEls.swipe.addEventListener('touchend', (e) => { if (sx === null) return; const dx = e.changedTouches[0].clientX - sx; sx = null; if (Math.abs(dx) > 50) (dx < 0 ? next : prev)(); });
  const onHash = () => { if (writingHash) return; const m = /stage=([\w-]+)/.exec(location.hash); if (m) goTo(m[1], { fromHash: true }); };
  addEventListener('hashchange', onHash);

  /** Per-frame: ease reveals, glide camera, tick active + still-visible handlers. */
  function update(dt) {
    state.dt = dt; state.time += dt;
    if (pending && (!glide || glide.t >= LAG_T)) commit(pending.e, false);
    const tgt = targets(), k = 1 - Math.exp(-dt * 4.5);
    const before = { ...state.reveal };
    for (const p of ALL_PARTS) {
      const t = tgt.has(p) ? 1 : 0, r = state.reveal[p];
      state.reveal[p] = state.reducedMotion || Math.abs(t - r) < 0.002 ? t : r + (t - r) * k;
    }
    if (glide) {
      glide.t += dt / glide.dur; const x = Math.min(1, glide.t), e = easeInOutCubic(x);
      camera.position.lerpVectors(glide.p0, glide.p1, e); controls.target.lerpVectors(glide.t0, glide.t1, e);
      if (glide.arc > 0) { const bow = Math.sin(Math.PI * x) * glide.arc; camera.position.addScaledVector(_v.subVectors(camera.position, controls.target).normalize(), bow); }
      if (glide.t >= 1) glide = null;
    }
    controls.update();
    STAGES.forEach((st, i) => {
      const h = handlerOf(st); if (!h?.update) return;
      // tick while current, still visible, or changed this frame (so the final reveal=0 is delivered and parts hide)
      if (force > 0 || i === state.stageIndex || st.parts.some((p) => state.reveal[p] > 0.001 || state.reveal[p] !== before[p])) h.update(dt, state);
    });
    if (force > 0) force--;
  }
  const _v = new ctx.THREE.Vector3();
  const m0 = /stage=([\w-]+)/.exec(location.hash);
  goTo(m0 && stageIndexOf(m0[1]) >= 0 ? m0[1] : 0, { instant: !!m0, fromHash: !!m0 });
  return { goTo, next, prev, update, get index() { return slide; }, get from() { return from; }, get settled() { return !pending && !glide; }, dispose() { removeEventListener('keydown', onKey); removeEventListener('hashchange', onHash); removeEventListener('resize', onResize); } };
}
