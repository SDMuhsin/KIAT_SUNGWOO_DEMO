// sage.js - module "sage": stage 12 (sage-idea), 13 (sage-overhead), 14 (sage-jax), 15 (sage-results). See llmdocs/CONTRACT_NOTES.md "sage API".
// Each stage lives in its own far-away zone (sage_kit ZONES). Numbers: model.FACTS (decks) or MODEL ESTIMATES (sage_math.js); layout illustrative.
import { STAGES } from '../stage.js';
import { createIdea } from './sage_idea.js';
import { createOverhead } from './sage_overhead.js';
import { createJax } from './sage_jax.js';
import { createResults } from './sage_results.js';
import * as PM from './physics_math.js';

export default function createSage(ctx) {
  const { THREE, util, theme } = ctx;
  const group = new THREE.Group(); group.name = 'sage';
  const at = (id) => STAGES.findIndex((s) => s.id === id);
  const shared = { idx: { cost: at('autograd-cost'), idea: at('sage-idea'), overhead: at('sage-overhead'), jax: at('sage-jax'), results: at('sage-results') }, blend: 0, mem: () => ctx.modules?.memory?.api };
  const idea = createIdea(ctx, shared), overhead = createOverhead(ctx, shared), jax = createJax(ctx, shared), results = createResults(ctx, shared);
  group.add(idea.group, overhead.group, jax.group, results.group);

  // The memory scene keeps the AD-graph block (set by physics at stage 11). Stage 13 shrinks it with the blend (to ~0), later stages drop it and release the hold.
  const M = { key: '', held: null, touched: false };
  function syncMemory(state) {
    const i = state.stageIndex, ix = shared.idx, m = shared.mem(); if (!m || i < ix.cost) return;
    const cfg = ctx.params(), bl = i <= ix.idea ? 0 : i === ix.overhead ? shared.blend : 1, hold = i <= ix.overhead;
    const key = [hold, Math.round(bl * 100), cfg.batch, cfg.width, cfg.depth, cfg.precision, cfg.pdeOrder, cfg.nResiduals].join('|');
    if (key === M.key) return; M.key = key; M.touched = true;
    if (M.held !== hold) { M.held = hold; m.hold('physics', hold); }
    if (!hold || bl >= 0.995) m.setExtraBlock('autograd', null);
    else m.setExtraBlock('autograd', { label: 'AUTOGRAD GRAPH', bytes: Math.max(1, PM.costRatios(cfg).graphBytes * (1 - bl)), color: theme.COLORS.physics });
  }
  const mk = (part) => ({ enter() { part.enter?.(); }, exit() {},
    update(dt, state) { part.update(dt, state, state.stageId === part.stageId); syncMemory(state); },
    get camera() { return part.frame(); } });
  idea.stageId = 'sage-idea'; overhead.stageId = 'sage-overhead'; jax.stageId = 'sage-jax'; results.stageId = 'sage-results';
  const stages = { 'sage-idea': mk(idea), 'sage-overhead': mk(overhead), 'sage-jax': mk(jax), 'sage-results': mk(results) };
  const api = { idea: idea.api, overhead: overhead.api, jax: jax.api, results: results.api, createPhysicsGraph: ctx.modules?.physics?.api?.createPhysicsGraph };
  return { group, stages, api,
    dispose() { idea.dispose(); overhead.dispose(); jax.dispose(); results.dispose(); const m = shared.mem(); if (M.touched) { m?.hold('physics', false); m?.setExtraBlock('autograd', null); } util.disposeObject(group); } };
}
