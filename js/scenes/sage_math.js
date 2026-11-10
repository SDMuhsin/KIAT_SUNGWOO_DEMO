// sage_math.js - pure helpers for the SAGE scenes (stages 12-15). No THREE, no DOM: runs in node (tests/sage.test.mjs).
// Facts come from model.FACTS (Sep/Oct 2026 decks). Everything computed by the cost model is a MODEL ESTIMATE.
// The 40-entry tape STRUCTURE (count and op mix) follows SAGE_EXPLAINED.tex; the individual entries are ILLUSTRATIVE.
import * as M from '../model.js';
import * as MM from './memory_math.js';
import * as PM from './physics_math.js';

const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const ss = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
export { lerp, clamp, ss };

// ------------------------------------------------------------------ stage 12: tape, VJP rules, generated code
/** Op colours (hex numbers) = chip colours = tape-cell colours. */
export const OPCOL = { matmul: 0x2fd0e6, mul: 0xf2c14e, add: 0x3ab795, sub: 0x7fd6a8, scale: 0xc7a6ff, pow2: 0x9b6bd6, sqrt: 0xe0408a };
/** The VJP rules shown as chips (SAGE_EXPLAINED.tex table; the real file has 10 rules). */
export const RULES = [
  { op: 'matmul', name: 'matmul', rule: 'transpose', math: 'x̄ += Dᵀ ȳ' },
  { op: 'mul', name: 'mul', rule: 'swap operands', math: 'ā += ȳ·b,  b̄ += ȳ·a' },
  { op: 'add', name: 'add', rule: 'copy', math: 'ā += ȳ,  b̄ += ȳ' },
  { op: 'sub', name: 'sub', rule: 'negate one side', math: 'ā += ȳ,  b̄ -= ȳ' },
  { op: 'scale', name: 'scale', rule: 'scale by constant', math: 'x̄ += c·ȳ' },
  { op: 'pow2', name: 'square', rule: 'power rule', math: 'x̄ += 2x·ȳ' },
  { op: 'sqrt', name: 'sqrt', rule: 'chain rule', math: 'x̄ += ȳ / (2√x)' },
];
export const ruleIndex = (op) => RULES.findIndex((r) => r.op === op);

// [op, a, b?] ; entry i defines t{i}. Navier-Stokes + Smagorinsky flavoured, simplified. Exactly 40 entries.
const SPEC = [
  ['matmul', 'Dx', 'u'], ['matmul', 'Dy', 'u'], ['matmul', 'Dx', 'v'], ['matmul', 'Dy', 'v'], ['matmul', 'Dx', 'p'], ['matmul', 'Dy', 'p'], // t0-5 first derivatives
  ['add', 't0', 't3'],                                                                                                                  // t6 continuity
  ['matmul', 'Dx', 't0'], ['matmul', 'Dy', 't1'], ['matmul', 'Dx', 't2'], ['matmul', 'Dy', 't3'],                                       // t7-10 second derivatives
  ['mul', 'u', 't0'], ['mul', 'v', 't1'], ['add', 't11', 't12'], ['mul', 'u', 't2'], ['mul', 'v', 't3'], ['add', 't14', 't15'],         // t11-16 convection
  ['pow2', 't0'], ['pow2', 't3'], ['add', 't1', 't2'], ['pow2', 't19'], ['add', 't17', 't18'], ['scale', 't21', '2'], ['add', 't22', 't20'], ['sqrt', 't23'], ['scale', 't24', 'Cs2'], // t17-25 strain, |S|, nu_t
  ['mul', 't25', 't0'], ['matmul', 'Dx', 't26'], ['mul', 't25', 't1'], ['matmul', 'Dy', 't28'],                                          // t26-29 turbulent stress
  ['add', 't7', 't8'], ['scale', 't30', 'nu'], ['add', 't27', 't29'], ['add', 't31', 't32'],                                              // t30-33 diffusion
  ['add', 't13', 't4'], ['sub', 't34', 't33'], ['add', 't16', 't5'], ['sub', 't36', 't33'],                                               // t34-37 momentum residuals
  ['pow2', 't35'], ['pow2', 't37'],                                                                                                      // t38-39 squared residuals (loss terms)
];
export const TAPE_LEN = SPEC.length;
const isMat = (s) => s === 'Dx' || s === 'Dy';
/** The tape: [{i, op, a, b, out, text}] where text is the forward statement. */
export function buildTape() {
  return SPEC.map(([op, a, b], i) => {
    const A = isMat(a) ? `g['${a}']` : a;
    const text = op === 'matmul' ? `t${i} = ${A} @ ${b}` : op === 'mul' ? `t${i} = ${a} * ${b}` : op === 'add' ? `t${i} = ${a} + ${b}` : op === 'sub' ? `t${i} = ${a} - ${b}`
      : op === 'scale' ? `t${i} = ${b} * ${a}` : op === 'pow2' ? `t${i} = ${a} ** 2` : `t${i} = sqrt(${a})`;
    return { i, op, a, b: op === 'pow2' || op === 'sqrt' ? null : b, out: `t${i}`, text };
  });
}
/** VJP statements (adjoint accumulation as code strings) for one tape entry. */
export function vjpLines(e) {
  const adj = (n) => (isMat(n) || /^\d|^Cs2$|^nu$/.test(n) ? null : `adj_${n}`);
  const y = `adj_t${e.i}`, L = [];
  const push = (n, rhs, sign = '+') => { const t = adj(n); if (t) L.push(`${t} ${sign}= ${rhs}`); };
  switch (e.op) {
    case 'matmul': push(e.b, `g['${e.a}T'] @ ${y}`); break;
    case 'mul': push(e.a, `${y} * ${e.b}`); push(e.b, `${y} * ${e.a}`); break;
    case 'add': push(e.a, y); push(e.b, y); break;
    case 'sub': push(e.a, y); push(e.b, y, '-'); break;
    case 'scale': push(e.a, `${e.b} * ${y}`); break;
    case 'pow2': push(e.a, `2 * ${e.a} * ${y}`); break;
    case 'sqrt': push(e.a, `${y} / (2 * t${e.i})`); break;
    default: break;
  }
  return L;
}
/** Reverse walk: entries 39..0 with the rule that applies and its adjoint lines. */
export function reverseWalk(tape = buildTape()) {
  return tape.slice().reverse().map((e) => ({ i: e.i, op: e.op, rule: RULES[ruleIndex(e.op)].rule, lines: vjpLines(e) }));
}
/** The emitted function as lines (header, forward replay, seeds, backward, return). */
export function genCode(tape = buildTape()) {
  const L = ['def generated_backward(pred, g):', '  u, v, p = pred[:, 0], pred[:, 1], pred[:, 2]', '  # forward replay (fixed structure)'];
  for (const e of tape) L.push('  ' + e.text);
  L.push('  # adjoint seeds (MSE derivative)', '  adj_t38 = mask / M', '  adj_t39 = mask / M', '  # reverse sweep, accumulated adjoints');
  for (const r of reverseWalk(tape)) for (const l of r.lines) L.push('  ' + l);
  L.push('  return cat([adj_u, adj_v, adj_p])   # = v = dL/du_hat');
  return L;
}

/** Timeline of the stage-12 animation (seconds). */
export const TL = { split: [0, 3], trace: [3, 7.5], reverse: [7.5, 14], emit: [14, 17.5], compile: [17.5, 19.5], loop: 19.5 };
export const TL_END = TL.loop + 6;
export const DECK_CALLS = 30000; // "generated once, then called 30,000 times during training" (SAGE_EXPLAINED.tex example)
/** State of the stage-12 animation at clock t. */
export function ideaState(t, n = TAPE_LEN, code = 100) {
  const p = (r) => clamp((t - r[0]) / (r[1] - r[0]));
  const trace = p(TL.trace), rev = p(TL.reverse), emit = p(TL.emit), comp = p(TL.compile);
  const phase = t < TL.trace[0] ? 'split' : t < TL.reverse[0] ? 'trace' : t < TL.emit[0] ? 'reverse' : t < TL.compile[0] ? 'emit' : t < TL.loop ? 'compile' : 'loop';
  const done = Math.floor(rev * n + 1e-9);                       // entries already walked backwards
  return { phase, split: p(TL.split), traced: trace * n, tracedCount: Math.floor(trace * n + 1e-9), reverse: rev, cursor: phase === 'reverse' ? clamp(n - 1 - done, 0, n - 1) : -1,
    walked: done, emit, emitLines: Math.floor(emit * code), compile: comp, compiled: t >= TL.loop ? 1 : comp, loop: Math.max(0, t - TL.loop),
    calls: t < TL.loop ? 0 : Math.round(DECK_CALLS * ss(0, 5, t - TL.loop)) };
}

// ------------------------------------------------------------------ stage 13: before -> after blend (model estimates)
/** Everything the stage-13 visuals show for blend b in 0..1 (0 = PyTorch autograd graph, 1 = SAGE). */
export function overheadState(cfg, b) {
  const c = M.normalize(cfg), P = M.autogradGraph(c, 'physics'), S = M.autogradGraph(c, 'sage');
  const sp = PM.stepSlabs(c, 'physics'), ss_ = PM.stepSlabs(c, 'sage'), R = PM.costRatios(c);
  const slabs = ['fwd', 'bwd', 'opt', 'build', 'traverse'].map((k) => lerp(sp[k], ss_[k], b));
  const total = slabs.reduce((a, x) => a + x, 0);
  return { blend: b, adNodes: Math.round(lerp(P.totalNodes, 0, b)), netNodes: S.totalNodes, totalNodes: Math.round(lerp(P.totalNodes, S.totalNodes, b)),
    matmuls: Math.round(lerp(P.matmuls, S.matmuls, b)), matmulsBefore: P.matmuls, matmulsAfter: S.matmuls,
    memBytes: R.graphBytes * (1 - b), memBytesFull: R.graphBytes, slabs, slabsBefore: [sp.fwd, sp.bwd, sp.opt, sp.build, sp.traverse], total, totalBefore: sp.total, totalAfter: ss_.total,
    speedup: sp.total / ss_.total, graphOverheadBefore: sp.build + sp.traverse, graphOverheadAfter: ss_.build + ss_.traverse };
}

// ------------------------------------------------------------------ stage 14: compiled step, bf16 storage, single pass
/** Deck-measured series (Sep 2026): none, + half-width storage, + single-pass bookkeeping. */
export const SEP_SERIES = [
  { id: 'aug', label: 'August engine', ms: M.FACTS.stepMs.baseline, units: M.FACTS.traffic.baseline },
  { id: 'half', label: '+ half-width storage', ms: 9.9, units: null },
  { id: 'single', label: '+ single-pass bookkeeping', ms: M.FACTS.stepMs.sage, units: M.FACTS.traffic.sage },
];
/** Step ms + traffic for a toggle state; `measured` is true ONLY for the three exact deck figures (12.7 / 9.9 / 7.7 ms). */
export function jaxSeries({ sageHalf = false, sageSinglePass = false } = {}) {
  const tr = M.sageTraffic({ sageHalf, sageSinglePass });
  let ms = tr.stepMs, measured = false, idx = -1;
  if (!sageHalf && !sageSinglePass) { ms = SEP_SERIES[0].ms; measured = true; idx = 0; }
  else if (sageHalf && !sageSinglePass) { ms = SEP_SERIES[1].ms; measured = true; idx = 1; }
  else if (sageHalf && sageSinglePass) { ms = SEP_SERIES[2].ms; measured = true; idx = 2; }
  const trafficMeasured = (!sageHalf && !sageSinglePass) || (sageHalf && sageSinglePass);
  return { ms, measured, idx, units: tr.trafficUnits, trafficMeasured };
}
/** Fraction of the step that is network + derivative evaluation at the physics points (Sep profiling) and the bf16 cut (a fifth, not a half). */
export const SEP_PROFILE = { netDerivShare: 0.75, bf16Cut: 0.2, naiveCut: 0.5 };

/**
 * Model-estimate roofline ops of the SAGE step for toggles {sageCompile, sageHalf, sageSinglePass}.
 * netDeriv = network + derivative evaluation at the physics points (physics forward tower: its bytes scale with the deck's traffic
 * units, units/126), then the clean backward, activation kernels and the optimizer. Launch overhead (3 us/kernel assumed) disappears when compiled.
 */
export function jaxState(cfg, tog = {}) {
  const c = M.normalize(cfg), s = jaxSeries(tog), compile = !!tog.sageCompile;
  const sage = MM.opTable(c, 'sage'), phys = M.stepCost(c, 'physics');
  const mk = (id, name, flops, bytes) => {
    const tC = flops / (c.peakTflops * 1e12), tM = bytes / (c.bandwidthGBs * 1e9), time = Math.max(tC, tM);
    return { id, name, flops, bytes, intensity: flops / bytes, tCompute: tC, tMemory: tM, time, bound: tM >= tC ? 'memory' : 'compute', perf: time > 0 ? flops / time : 0 };
  };
  const scale = s.units / M.FACTS.traffic.baseline;
  const nd0 = mk('netDeriv', 'network + derivative eval', phys.fwd.flops, phys.fwd.bytes);
  const nd = mk('netDeriv', 'network + derivative eval', phys.fwd.flops, phys.fwd.bytes * scale);
  const rest = sage.ops.filter((o) => o.id === 'mmBwd' || o.id === 'act' || o.id === 'opt');
  const ops = [nd, ...rest], launches = sage.launches + 6; // + a few launches for the tower kernels
  const overhead = compile ? 0 : launches * MM.ASSUMPTIONS.launchUs * 1e-6, overheadBefore = launches * MM.ASSUMPTIONS.launchUs * 1e-6;
  const busy = ops.reduce((a, o) => a + o.time, 0), total = busy + overhead, totalAllOff = nd0.time + rest.reduce((a, o) => a + o.time, 0) + overheadBefore;
  return { ops, base: nd0, netDerivShare: nd.time / total, overhead, overheadShare: overhead / total, busy, total, totalAllOff, launches: compile ? 1 : launches,
    ridge: M.ridge(c), series: s, speedupVsAllOff: totalAllOff / total, bound: nd.bound, compile };
}

// ------------------------------------------------------------------ stage 15: results + Amdahl
const F = M.FACTS;
/** Headline strings, each derived from model.FACTS (the test asserts they equal the SPEC). */
export function headline() {
  const r = M.results();
  return { perStep: Math.round(r.perStep) + 'x', a40: Math.round(r.fineTuneA40) + 'x', rtx: '~' + Math.round(r.fineTune5080) + 'x', bedrock: '~' + Math.round(r.bedrockProjected / 10) * 10 + 'x',
    pytorchMs: F.perStepMs.pytorchUofW, oursMs: F.perStepMs.ours };
}
/**
 * Whole-run rows. setup is NOT accelerated, so PyTorch = same setup + (run - setup) steps. Setup/steps of OUR run from the deck bars
 * (fine-tune 62 s setup + 32 s steps; bedrock 12 min setup + 68 min steps). The RTX 5080 row has no reported split.
 */
export function runRows() {
  const A = F.fineTune.a40, R5 = F.fineTune.rtx5080, B = F.bedrock;
  return [
    { id: 'a40', name: 'Fine-tune, A40 (USask)', unit: 'min', before: A.beforeMin, after: A.afterMin, setup: 62 / 60, stepsAfter: 32 / 60, speed: M.results().fineTuneA40, tag: 'measured, Oct 2026 deck',
      labelBefore: `${A.beforeMin} min`, labelAfter: `${A.afterMin} min`, headline: Math.round(M.results().fineTuneA40) + 'x' },
    { id: 'rtx', name: 'Fine-tune, RTX 5080 (UofW)', unit: 'min', before: (R5.beforeMin[0] + R5.beforeMin[1]) / 2, after: R5.afterMin[0] ?? R5.afterMin, setup: null, stepsAfter: null, speed: M.results().fineTune5080,
      tag: 'measured by UofW', labelBefore: `${R5.beforeMin[0]}–${R5.beforeMin[1]} min`, labelAfter: '~1 min', headline: '~' + Math.round(M.results().fineTune5080) + 'x' },
    { id: 'bed', name: 'Bedrock, 600k steps, A40', unit: 'h', before: B.beforeH, after: B.afterH, setup: 12 / 60, stepsAfter: 68 / 60, speed: M.results().bedrockProjected, tag: 'PROJECTED',
      labelBefore: `~${B.beforeH} h`, labelAfter: `~${B.afterH} h`, headline: '~' + Math.round(M.results().bedrockProjected / 10) * 10 + 'x', projected: true },
  ];
}
/** Amdahl: s = setup share of the ACCELERATED run, k = per-step speedup; whole-run speedup = s + (1-s)k (setup identical on both). */
export const amdahl = (s, k) => s + (1 - s) * k;
export const setupShareOf = (row) => (row.setup == null ? null : row.setup / (row.setup + row.stepsAfter));
export function amdahlBars(s, k) {
  const before = s + (1 - s) * k;
  return { speedup: before, ours: { setup: s / before, steps: (1 - s) / before }, pytorch: { setup: s / before, steps: ((1 - s) * k) / before } };
}
export const PRESET_SHARE = { fine: 62 / 94, bedrock: 12 / 80 }; // two thirds, 15%
