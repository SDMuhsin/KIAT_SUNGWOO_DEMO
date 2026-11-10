// physics.js - module "physics": stage 8 (pinn), 9 (derivs), 10 (phys-grad), 11 (autograd-cost).
// Visuals are model-driven (js/model.js via physics_math.js); layout/colours are ILLUSTRATIVE. Physics API for other scenes
// (SAGE reuses the graph helper): ctx.modules.physics.api, see llmdocs/CONTRACT_NOTES.md "physics API".
import { STAGES } from '../stage.js';
import { createPinn } from './physics_pinn.js';
import { createDerivs } from './physics_derivs.js';
import { createGrad } from './physics_grad.js';
import { createCost } from './physics_cost.js';
import { createPhysicsGraph } from './physics_graph.js';
export { createPhysicsGraph };

export default function createPhysics(ctx) {
  const { THREE, util } = ctx;
  const group = new THREE.Group(); group.name = 'physics';
  const at = (id) => STAGES.findIndex((s) => s.id === id);
  const shared = {
    idx: { pinn: at('pinn'), derivs: at('derivs'), physgrad: at('phys-grad'), cost: at('autograd-cost') },
    net: () => ctx.modules?.network?.api, geo: () => ctx.modules?.geometry?.api, mem: () => ctx.modules?.memory?.api,
  };
  const pinn = createPinn(ctx, shared), derivs = createDerivs(ctx, shared), grad = createGrad(ctx, shared), cost = createCost(ctx, shared);
  group.add(pinn.group, derivs.group, grad.group, cost.group);
  const cur = (state, id) => state.stageId === id;
  const mk = (id, part) => ({
    enter() { part.enter?.(); }, exit() {},
    update(dt, state) { part.update(dt, state, cur(state, id)); },
    get camera() { return part.frame(); },
  });
  const stages = { pinn: mk('pinn', pinn), derivs: mk('derivs', derivs), 'phys-grad': mk('phys-grad', grad), 'autograd-cost': mk('autograd-cost', cost) };
  const api = { pinn: pinn.api, derivs: derivs.api, grad: grad.api, cost: cost.api, createPhysicsGraph, graphStats: () => grad.api.graphStats() };
  return { group, stages, api, dispose() { pinn.dispose(); derivs.dispose(); grad.dispose(); cost.dispose(); util.disposeObject(group); } };
}
