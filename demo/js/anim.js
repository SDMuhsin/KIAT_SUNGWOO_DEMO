// anim.js - pure timing for ONE training step (no THREE, node-testable). All durations are ILLUSTRATIVE seconds.
// Phases: points fly to the network -> forward through the network -> output + loss -> backward sweep through the autograd graphs
// (the physics graph takes longer the heavier it is) -> gradient reaches the network -> optimizer update.
import { clamp, lerp } from './util.js';

export const SEG = { pts: 1.0, vp: 0.9, net: 1.2, out: 0.6, dataBwd: 1.2, netBwd: 0.7, opt: 0.8 };

/** ends = model.sageEndpoints(cfg); e = eased SAGE progress 0..1. Returns windows in u-space (0..1) plus total seconds. */
export function timeline(ends, e) {
  const ratio = ends.base.bwdTime / Math.max(ends.sage.bwdTime, 1e-12);
  const physBase = SEG.dataBwd * clamp(Math.sqrt(ratio), 1, 4.5);
  const phys = lerp(physBase, SEG.dataBwd, e);
  const bwd = Math.max(SEG.dataBwd, phys);
  const vp = SEG.vp * clamp((e - 0.05) / 0.3); // SAGE-only extra no-grad pass that produces v (0 at baseline, so baseline timing is unchanged)
  const total = SEG.pts + vp + SEG.net + SEG.out + bwd + SEG.netBwd + SEG.opt + 0.5; // 0.5 s hold at the end
  const w = (a, d) => [a / total, (a + d) / total];
  const t1 = SEG.pts + vp, t2 = t1 + SEG.net, t3 = t2 + SEG.out, t4 = t3 + bwd, t5 = t4 + SEG.netBwd;
  return { total, pts: w(0, SEG.pts), vp: w(SEG.pts, vp), hasVp: vp > 1e-6, fwd: w(t1, SEG.net), out: w(t2, SEG.out), bwdData: w(t3, SEG.dataBwd), bwdPhys: w(t3, phys), netBwd: w(t4, SEG.netBwd), opt: w(t5, SEG.opt) };
}

/** progress 0..1 of u inside window [a,b] (clamped). */
export const prog = (u, win) => clamp((u - win[0]) / (win[1] - win[0]));

export function phaseAt(u, tl, e) {
  if (u < tl.pts[1]) return 'points travel to the network';
  if (tl.hasVp && u >= tl.vp[0] && u < tl.vp[1]) return 'extra cheap pass, no graph: computes v';
  if (u < tl.fwd[1]) return 'forward pass';
  if (u < tl.out[1]) return 'output and loss';
  if (u < tl.netBwd[0]) return e > 0.5 ? 'backward: one ordinary pass (v from the lane)' : 'backward: through the autograd graphs';
  if (u < tl.opt[0]) return 'gradients reach the network';
  if (u < tl.opt[1]) return 'optimizer update';
  return 'step done';
}
