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
  { id: 'geometry', title: '1. The geometry', module: 'geometry', parts: ['geometry.duct', 'geometry.fieldslot'], knobs: [], badges: ['illustrative'],
    caption: 'A battery-pack-like cooling duct with inlet and outlet. Beside it, an empty slot for the field the network will predict.', camera: { pos: [0, 5, 16], target: [0, 0, 0] } },
  { id: 'sampling', title: '2. Sampling points', module: 'geometry', parts: ['sampling.interior', 'sampling.boundary'], knobs: ['batch'], badges: ['illustrative'],
    caption: 'Interior collocation points and boundary points are sampled from the geometry. Each is an (x, y, z) input.', camera: { pos: [-2, 5, 14], target: [0, 0, 0] } },
  { id: 'network', title: '3. The network', module: 'network', parts: ['network.layers', 'network.edges'], knobs: ['width', 'depth', 'activation'], badges: ['illustrative'],
    caption: 'An MLP drawn in the same space: planes of neurons, edges with thickness proportional to |weight|.', camera: { pos: [4, 5, 18], target: [3, 0, 0] } },
  { id: 'forward', title: '4. Forward pass', module: 'network', parts: ['forward.pulse', 'forward.field'], knobs: ['batch', 'width', 'depth', 'activation'], badges: ['illustrative'],
    caption: 'Points travel through the layers; the output field (u, v, w, p) resolves on the geometry.', camera: { pos: [6, 4, 17], target: [4, 0, 0] } },
  { id: 'memory-fwd', title: '5. Memory: weights and activations', module: 'memory', parts: ['memory.weights', 'memory.activations'], knobs: ['batch', 'width', 'depth', 'precision'], badges: ['model estimate'],
    caption: 'Memory blocks appear: WEIGHTS and ACTIVATIONS stacked per layer, volume proportional to bytes.', camera: { pos: [4, 6, 18], target: [3, -1, 0] } },
  { id: 'backward', title: '6. Backward pass', module: 'memory', parts: ['backward.flow', 'memory.gradients', 'memory.optimizer'], knobs: ['batch', 'width', 'depth', 'precision'], badges: ['model estimate'],
    caption: 'Autograd walks output to input, consuming activations; GRADIENTS appear, and the Adam OPTIMIZER STATE (m, v) joins them.', camera: { pos: [5, 6, 19], target: [3, -1, 0] } },
  { id: 'bounds', title: '7. Compute- vs memory-bound', module: 'memory', parts: ['bounds.overlay'], knobs: ['batch', 'width', 'depth', 'precision', 'bandwidth', 'peakTflops'], badges: ['model estimate'],
    caption: 'Roofline view: each block coloured by arithmetic intensity against the ridge point; move the knobs and watch bounds flip.', camera: { pos: [3, 8, 20], target: [3, -1, 0] } },
  { id: 'pinn', title: '8. Physics constraints (PINN)', module: 'physics', parts: ['pinn.constraints', 'pinn.residualfield'], knobs: ['nResiduals', 'lambda'], badges: ['illustrative'],
    caption: 'The PDE residual (continuity, momentum, k-epsilon) becomes extra loss: the field is violated, then satisfied.', camera: { pos: [-3, 5, 16], target: [0, 0, 0] } },
  { id: 'derivs', title: '9. Derivatives w.r.t. inputs', module: 'physics', parts: ['derivs.tower'], knobs: ['pdeOrder', 'nResiduals', 'depth'], badges: ['illustrative', 'model estimate'],
    caption: 'The residual needs u_x, u_xx, ... of the network with respect to its INPUTS: a derivative tower multiplying memory and compute.', camera: { pos: [-1, 6, 15], target: [1, 1, 0] } },
  { id: 'phys-grad', title: '10. Weight gradient vs physics gradient', module: 'physics', parts: ['physgrad.weightpath', 'physgrad.physpath'], knobs: ['pdeOrder', 'nResiduals', 'depth'], badges: ['illustrative', 'model estimate'],
    caption: 'Left: the weight gradient, one simple chain. Right: the physics-loss gradient differentiates through derivatives of the network, a branching second-order graph.', camera: { pos: [0, 4, 15], target: [0, 0, 0] } },
  { id: 'autograd-cost', title: '11. Autograd cost of physics', module: 'physics', parts: ['autograd.graph', 'autograd.cost'], knobs: ['pdeOrder', 'nResiduals', 'depth', 'batch'], badges: ['model estimate'],
    caption: 'The full autograd graph for the physics loss: nodes, stored intermediates, traversals; memory and time inflate versus plain supervised training.', camera: { pos: [2, 6, 17], target: [1, 0, 0] } },
  { id: 'sage-idea', title: '12. SAGE: split the gradient', module: 'sage', parts: ['sage.split', 'sage.tape'], knobs: [], badges: ['illustrative'],
    caption: 'v = dL/d(u-hat) is PDE-only and closed form, derived symbolically ONCE (trace, tape, reverse-walk VJP, emit code); then one clean backprop through the network Jacobian.', camera: { pos: [0, 4, 15], target: [0, 0, 0] } },
  { id: 'sage-overhead', title: '13. SAGE: overhead removed', module: 'sage', parts: ['sage.before_after'], knobs: ['sageBlend', 'batch', 'width', 'depth', 'pdeOrder', 'nResiduals', 'bandwidth'], badges: ['model estimate'],
    caption: 'Before and after: the autograd graph vanishes, 24 matmuls become 10, no AD-graph memory.', camera: { pos: [0, 5, 17], target: [0, 0, 0] } },
  { id: 'sage-jax', title: '14. SAGE: one compiled program', module: 'sage', parts: ['sage.jax'], knobs: ['sageCompile', 'sageHalf', 'sageSinglePass', 'bandwidth', 'peakTflops', 'precision'], badges: ['model estimate', 'illustrative'],
    caption: 'A single compiled GPU program plus half-width storage and single-pass bookkeeping: traffic 126 to 31 units, 12.7 to 7.7 ms/step (Sep profiling).', camera: { pos: [2, 4, 15], target: [1, 0, 0] } },
  { id: 'sage-results', title: '15. Results, honestly', module: 'sage', parts: ['sage.results'], knobs: ['showProjected'], badges: ['projected'],
    caption: '50x per step vs PyTorch as UofW runs it; whole fine-tune run 10-16x; bedrock ~40x is PROJECTED, not measured.', camera: { pos: [0, 3, 14], target: [0, 0, 0] } },
];

/** Cumulative set of revealed part ids at stage index i. */
export function partsAt(i) {
  const s = new Set();
  for (let k = 0; k <= i; k++) { STAGES[k].parts.forEach((p) => s.add(p)); (STAGES[k].hide ?? []).forEach((p) => s.delete(p)); }
  return s;
}
export const ALL_PARTS = [...new Set(STAGES.flatMap((s) => s.parts))];
export const stageIndexOf = (id) => STAGES.findIndex((s) => s.id === id);

/**
 * createStageController({ctx, modules, hudEls}) - owns state.reveal, camera glide, HUD, hash, keys, swipe.
 * modules: { [moduleId]: {group, stages:{[stageId]:handler}, dispose} }.  Returns {goTo,next,prev,update,index,dispose}.
 */
export function createStageController({ ctx, modules, hudEls }) {
  const { state, camera, controls, knobs, hud } = ctx;
  state.reveal = Object.fromEntries(ALL_PARTS.map((p) => [p, 0]));
  state.stageIndex = -1; state.stageId = null; state.prevIndex = -1; state.time = 0; state.dt = 0;
  let force = 0, glide = null, active = null, writingHash = false;
  const handlerOf = (st) => modules[st.module]?.stages?.[st.id];
  const targets = () => partsAt(state.stageIndex);

  function cameraFor(st) { return handlerOf(st)?.camera ?? st.camera; }
  function startGlide(cam, instant) {
    if (instant || state.reducedMotion) { camera.position.set(...cam.pos); controls.target.set(...cam.target); controls.update(); glide = null; return; }
    glide = { t: 0, dur: 1.4, p0: camera.position.clone(), t0: controls.target.clone(), p1: camera.position.clone().set(...cam.pos), t1: controls.target.clone().set(...cam.target) };
  }
  let moved = false, rzT = 0;
  controls.addEventListener('start', () => { glide = null; moved = true; });
  // re-frame on window resize (only if the user has not orbited away since entering this stage)
  const onResize = () => { clearTimeout(rzT); rzT = setTimeout(() => { if (!moved && state.stageIndex >= 0) startGlide(cameraFor(STAGES[state.stageIndex]), true); }, 120); };
  addEventListener('resize', onResize);

  function goTo(target, { instant = false, fromHash = false } = {}) {
    let i = typeof target === 'number' ? target : stageIndexOf(target);
    if (i < 0 || i >= STAGES.length) return false;
    if (i === state.stageIndex) return true;
    const st = STAGES[i];
    if (active) { try { active.exit?.(); } catch (e) { console.error('stage exit', e); } }
    state.prevIndex = state.stageIndex; state.stageIndex = i; state.stageId = st.id;
    hud.clearReadouts(); force = 2;
    const tgt = targets();
    if (instant || state.reducedMotion) for (const p of ALL_PARTS) state.reveal[p] = tgt.has(p) ? 1 : 0;
    moved = false; startGlide(cameraFor(st), instant);
    active = handlerOf(st) ?? null;
    knobs.setVisible(st.knobs);
    try { active?.enter?.(instant ? 1 : 0); } catch (e) { console.error('stage enter', e); }
    renderHud(st);
    if (!fromHash) { writingHash = true; history.replaceState(null, '', '#stage=' + st.id); setTimeout(() => (writingHash = false), 0); }
    return true;
  }
  const next = () => goTo(Math.min(STAGES.length - 1, state.stageIndex + 1));
  const prev = () => goTo(Math.max(0, state.stageIndex - 1));

  function renderHud(st) {
    const h = active;
    hudEls.title.textContent = h?.title ?? st.title; hudEls.caption.textContent = h?.caption ?? st.caption;
    hudEls.count.textContent = `${state.stageIndex + 1} / ${STAGES.length}`;
    hudEls.bar.style.width = ((state.stageIndex + 1) / STAGES.length * 100) + '%';
    hudEls.prev.disabled = state.stageIndex === 0; hudEls.next.disabled = state.stageIndex === STAGES.length - 1;
    hudEls.dots.querySelectorAll('button').forEach((b, k) => { b.setAttribute('aria-current', k === state.stageIndex ? 'step' : 'false'); b.classList.toggle('done', k < state.stageIndex); });
    hud.setBadges(st.badges ?? []);
    document.title = `SAGE demo - ${st.title}`;
  }

  // dots
  STAGES.forEach((st, k) => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'dot'; b.title = st.title; b.setAttribute('aria-label', 'Go to ' + st.title);
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
    const tgt = targets(), k = 1 - Math.exp(-dt * 4.5);
    const before = { ...state.reveal };
    for (const p of ALL_PARTS) {
      const t = tgt.has(p) ? 1 : 0, r = state.reveal[p];
      state.reveal[p] = state.reducedMotion || Math.abs(t - r) < 0.002 ? t : r + (t - r) * k;
    }
    if (glide) {
      glide.t += dt / glide.dur; const e = easeInOutCubic(Math.min(1, glide.t));
      camera.position.lerpVectors(glide.p0, glide.p1, e); controls.target.lerpVectors(glide.t0, glide.t1, e);
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
  const m0 = /stage=([\w-]+)/.exec(location.hash);
  goTo(m0 && stageIndexOf(m0[1]) >= 0 ? m0[1] : 0, { instant: !!m0, fromHash: !!m0 });
  return { goTo, next, prev, update, get index() { return state.stageIndex; }, dispose() { removeEventListener('keydown', onKey); removeEventListener('hashchange', onHash); removeEventListener('resize', onResize); } };
}
