// _placeholder.js - FOUNDATION PLACEHOLDER. Real scene modules REPLACE their file wholesale and must not import this.
// One labelled box per stage owned by `moduleId`, gated on that stage's first part, height reacting to the width knob
// (proves live knobs), and a model readout where relevant.
import { STAGES } from '../stage.js';

export function createPlaceholder(ctx, moduleId, colorKey) {
  const { THREE, theme, util, knobs, model, hud } = ctx;
  const group = new THREE.Group(); group.name = moduleId;
  const mine = STAGES.map((s, i) => ({ s, i })).filter(({ s }) => s.module === moduleId);
  const items = {};
  for (const { s, i } of mine) {
    const g = new THREE.Group(); g.position.set(((i % 5) - 2) * 3, 0, -Math.floor(i / 5) * 3); g.visible = false;
    const box = new THREE.Mesh(new THREE.BoxGeometry(1.6, 1.6, 1.6), theme.material(THREE, colorKey, { opacity: 0.85 }));
    const label = util.makeLabel(THREE, `${i + 1}:${s.id}`, { size: 0.3 });
    label.position.y = 1.6; g.add(box, label); group.add(g);
    items[s.id] = { g, box, part: s.parts[0] };
  }
  const apply = () => { const h = 0.4 + Math.log2(knobs.get('width') / 16) * 0.35; for (const k in items) items[k].box.scale.y = h; };
  knobs.on('width', apply); apply();

  const stages = {};
  for (const { s } of mine) {
    const it = items[s.id];
    stages[s.id] = {
      enter() { it.box.rotation.y = 0; },
      exit() {},
      update(dt, state) {
        util.gateGroup(it.g, util.revealOf(state, it.part));
        it.box.rotation.y += dt * (state.reducedMotion ? 0 : 0.6);
        if (state.stageId === s.id && model) {
          const c = model.stepCost(ctx.params()); hud.readout('t', 'Step time (model)', model.fmtTime(c.total.time), 'model estimate');
          hud.readout('b', 'Bound', c.total.bound);
        }
      },
    };
  }
  return { group, stages, dispose() { util.disposeObject(group); } };
}
