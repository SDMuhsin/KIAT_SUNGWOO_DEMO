// memory.js - module "memory": stage 5 (memory blocks), stage 6 (autograd backward + optimizer), stage 7 (compute/memory bounds).
// All sizes come from js/model.js via memory_math.js (MODEL ESTIMATE, never measured). Memory API for other scenes:
//   ctx.modules.memory.api  - see llmdocs/CONTRACT_NOTES.md "memory API".
import * as MM from './memory_math.js';
import { frameBox } from './memory_cam.js';
import { STAGES } from '../stage.js';

const MAXL = 14, NPAR = 8, NB = MAXL + NPAR + 1, I_PAR = MAXL, I_LEG = MAXL + NPAR;
const FLOOR = MM.MEM_FLOOR, PF = MM.PLOT.floorY, SHELF_Z0 = 3.5, SHELF_X0 = 0.3, LEG_X = -1.4;
const NP = 54;               // optimizer particles
const NRF = 48;              // roof curve points
const KEYS = ['weights', 'gradients', 'adamM', 'adamV'];
const BOUND_COL = { memory: 0xe0408a, compute: 0x2fd0e6 }, OVER_COL = 0x8a94a6;
// 12 box edges as corner-index pairs (corner bits: x=1,y=2,z=4)
const EDGES = []; for (let a = 0; a < 8; a++) for (let b = 1; b < 8; b <<= 1) if (!(a & b)) EDGES.push([a, a | b]);

export default function createMemory(ctx) {
  const { THREE, knobs, util, hud, model, theme } = ctx;
  const { clamp, smoothstep: ss } = util;
  const COL = theme.COLORS;
  const group = new THREE.Group(); group.name = 'memory'; group.visible = false;
  let disposed = false, lastStamp = -1;
  const net = () => ctx.modules?.network?.api;
  const boundsIdx = STAGES.findIndex((s) => s.id === 'bounds');

  // ---------------------------------------------------------------- GPU resources (fixed capacity)
  const boxGeo = new THREE.BoxGeometry(1, 1, 1), sphGeo = new THREE.SphereGeometry(1, 10, 8);
  const boxMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5, metalness: 0.05, transparent: true, opacity: 0.9 });
  const boxes = new THREE.InstancedMesh(boxGeo, boxMat, NB); boxes.frustumCulled = false; boxes.renderOrder = 2;
  boxes.setColorAt(0, new THREE.Color(1, 1, 1)); for (let i = 0; i < NB; i++) boxes.instanceMatrix.array[i * 16 + 15] = 1;
  const ol = (n) => { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 6), 3)); g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 6), 3));
    const l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false })); l.frustumCulled = false; l.renderOrder = 3; return l; };
  const outline = ol(NB * 12), links = ol(MAXL * 2 + 8);
  links.material.blending = THREE.AdditiveBlending;
  const partGeo = sphGeo, partMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95, depthWrite: false });
  const parts = new THREE.InstancedMesh(partGeo, partMat, NP); parts.frustumCulled = false; parts.renderOrder = 5; parts.setColorAt(0, new THREE.Color(1, 1, 1));
  for (let i = 0; i < NP; i++) parts.instanceMatrix.array[i * 16 + 15] = 1;
  // floor + grid
  const floorMat = new THREE.MeshBasicMaterial({ color: 0x141c2c, transparent: true, opacity: 0.55, depthWrite: false, side: THREE.DoubleSide });
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), floorMat); floor.rotation.x = -Math.PI / 2; floor.position.set(3.6, FLOOR - 0.02, 2.2); floor.scale.set(13.0, 11.2, 1); floor.renderOrder = 0;
  const gridG = new THREE.BufferGeometry(); { const p = []; for (let x = -2; x <= 10; x += 1) p.push(x - 0.5, FLOOR, -3.4, x - 0.5, FLOOR, 7.8); for (let z = -3; z <= 7; z += 1) p.push(-2.9, FLOOR, z + 0.4, 10, FLOOR, z + 0.4); gridG.setAttribute('position', new THREE.Float32BufferAttribute(p, 3)); }
  const gridMat = new THREE.LineBasicMaterial({ color: 0x26314a, transparent: true, opacity: 0.55 }); const grid = new THREE.LineSegments(gridG, gridMat);
  // wavefront
  const waveMat = (o) => new THREE.MeshBasicMaterial({ color: COL.gradients, transparent: true, opacity: o, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide });
  const planeGeo = new THREE.PlaneGeometry(1, 1), edgeGeo = new THREE.EdgesGeometry(planeGeo);
  const waves = [0.2, 0.11, 0.05].map((o) => { const m = new THREE.Mesh(planeGeo, waveMat(o)); m.rotation.y = Math.PI / 2; m.renderOrder = 4; m.userData.base = o; return m; });
  const waveEdge = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: 0xffc27a, transparent: true, opacity: 0.9 })); waves[0].add(waveEdge);
  // roofline plot objects
  const tmp0 = new THREE.Color(); const P = MM.PLOT, plot = new THREE.Group(); plot.visible = false;
  const zoneMat = (c) => new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.17, depthWrite: false, side: THREE.DoubleSide });
  const zoneM = new THREE.Mesh(planeGeo, zoneMat(BOUND_COL.memory)), zoneC = new THREE.Mesh(planeGeo, zoneMat(BOUND_COL.compute)), zoneO = new THREE.Mesh(planeGeo, zoneMat(OVER_COL));
  for (const z of [zoneM, zoneC, zoneO]) { z.rotation.x = -Math.PI / 2; z.renderOrder = 0; z.userData.base = 0.17; plot.add(z); }
  const zFront = P.laneZ0 + 0.55, zBack = MM.laneZ(5) - 0.5, zMid = (zFront + zBack) / 2, zLen = zFront - zBack;
  const pfMat = new THREE.MeshBasicMaterial({ color: 0x141c2c, transparent: true, opacity: 0.7, depthWrite: false, side: THREE.DoubleSide }); pfMat.userData.base = 0.7;
  const pfloor = new THREE.Mesh(planeGeo, pfMat); pfloor.rotation.x = -Math.PI / 2; pfloor.position.set((P.xL - 2.0 + P.xO + 1.0) / 2, PF - 0.01, zMid + 0.9); pfloor.scale.set(P.xO + 1.0 - P.xL + 2.0, zLen + 5.2, 1); plot.add(pfloor);
  const wallMat = new THREE.MeshBasicMaterial({ color: 0x101826, transparent: true, opacity: 0.6, depthWrite: false, side: THREE.DoubleSide }); wallMat.userData.base = 0.6;
  const wall = new THREE.Mesh(planeGeo, wallMat); wall.position.set((P.xL + P.xR) / 2, PF + P.wallH / 2, P.wallZ); wall.scale.set(P.xR - P.xL + 0.4, P.wallH + 0.4, 1); plot.add(wall);
  const plotLinesG = new THREE.BufferGeometry(); { const p = [];
    for (let d = P.aiLo; d <= P.aiHi; d++) { const x = MM.xOfAI(Math.pow(10, d)); p.push(x, PF + 0.01, zFront, x, PF + 0.01, zBack, x, PF, P.wallZ, x, PF + P.wallH, P.wallZ); }
    for (let d = P.perfLo + 1; d <= 14; d++) { const y = MM.yOfPerf(Math.pow(10, d)); p.push(P.xL, y, P.wallZ, P.xR, y, P.wallZ); }
    for (let k = 0; k <= 5; k++) { const z = MM.laneZ(k); p.push(P.xL - 1.9, PF + 0.01, z, P.xO + 0.9, PF + 0.01, z); }
    plotLinesG.setAttribute('position', new THREE.Float32BufferAttribute(p, 3)); }
  const plotLines = new THREE.LineSegments(plotLinesG, new THREE.LineBasicMaterial({ color: 0x33415f, transparent: true, opacity: 0.8 })); plotLines.material.userData.base = 0.8; plot.add(plotLines);
  // roof ribbon (thick line) with per-vertex colour
  const roofG = new THREE.BufferGeometry(); roofG.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NRF * 2 * 3), 3)); roofG.setAttribute('color', new THREE.BufferAttribute(new Float32Array(NRF * 2 * 3), 3));
  { const idx = []; for (let i = 0; i < NRF - 1; i++) { const a = i * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); } roofG.setIndex(idx); }
  const roofMat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 1 }); roofMat.userData.base = 1;
  const roof = new THREE.Mesh(roofG, roofMat); roof.frustumCulled = false; roof.renderOrder = 2; plot.add(roof);
  const ridgeG = new THREE.BufferGeometry(); ridgeG.setAttribute('position', new THREE.BufferAttribute(new Float32Array(4 * 3), 3));
  const ridgeLine = new THREE.LineSegments(ridgeG, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 })); ridgeLine.material.userData.base = 0.8; ridgeLine.frustumCulled = false; plot.add(ridgeLine);
  const dropG = new THREE.BufferGeometry(); dropG.setAttribute('position', new THREE.BufferAttribute(new Float32Array(5 * 2 * 3), 3));
  const dropLines = new THREE.LineSegments(dropG, new THREE.LineBasicMaterial({ color: 0x9fb0cc, transparent: true, opacity: 0.45 })); dropLines.material.userData.base = 0.45; dropLines.frustumCulled = false; plot.add(dropLines);
  const pillars = new THREE.InstancedMesh(boxGeo, boxMat.clone(), 9); pillars.frustumCulled = false; pillars.renderOrder = 2; pillars.setColorAt(0, new THREE.Color(1, 1, 1));
  for (let i = 0; i < 9; i++) pillars.instanceMatrix.array[i * 16 + 15] = 1; pillars.material.userData.base = 0.92; pillars.material.opacity = 0.92;
  const marks = new THREE.InstancedMesh(sphGeo, new THREE.MeshBasicMaterial({ color: 0xffffff }), 5); marks.frustumCulled = false; marks.renderOrder = 4; marks.setColorAt(0, new THREE.Color(1, 1, 1));
  for (let i = 0; i < 5; i++) marks.instanceMatrix.array[i * 16 + 15] = 1;
  const chips = new THREE.InstancedMesh(boxGeo, new THREE.MeshBasicMaterial({ color: 0xffffff }), 12); chips.frustumCulled = false; chips.setColorAt(0, new THREE.Color(1, 1, 1));
  plot.add(pillars, marks, chips);
  const CHIPS = [['w', 'a'], ['a', 'g', 'w'], ['a'], ['a'], ['w', 'g', 'm', 'v']]; const chipCol = { w: COL.weights, a: COL.activations, g: COL.gradients, m: COL.optimizer, v: 0x7e57c2 };
  { let k = 0; for (let i = 0; i < 5; i++) CHIPS[i].forEach((c, j) => { const o = k * 16; const a = chips.instanceMatrix.array; a[o] = a[o + 5] = a[o + 10] = 0.26; a[o + 12] = P.xL - 0.45 - j * 0.34; a[o + 13] = PF + 0.14; a[o + 14] = MM.laneZ(i); a[o + 15] = 1; tmp0.setHex(chipCol[c]); chips.setColorAt(k, tmp0); k++; }); chips.count = k; chips.instanceMatrix.needsUpdate = true; chips.instanceColor.needsUpdate = true; }
  group.add(floor, grid, boxes, outline, links, parts, ...waves, plot);

  // ---------------------------------------------------------------- labels
  let LS = 1; const labs = {};
  const mkLab = (key, size, color, bg) => { const s = util.makeLabel(THREE, ' ', { size, color, ...(bg ? { bg } : {}) }); s.userData.base = size; s.userData.ls = 1; s.visible = false; group.add(s); labs[key] = s; return s; };
  const setT = (s, text) => { if (s.userData.ls !== LS) { s.userData.opts.size = s.userData.base * LS; s.userData.ls = LS; s.userData.text = null; } util.setLabel(s, text); };
  const hex = (n) => '#' + n.toString(16).padStart(6, '0');
  const labSpec = [
    ['W', 0.34, hex(COL.weights)], ['Wv', 0.3, '#dbe8ff'], ['G', 0.34, hex(COL.gradients)], ['Gv', 0.3, '#ffe0c0'], ['M', 0.34, '#c9a8ef'], ['Mv', 0.3, '#e6d6fa'], ['V', 0.34, '#c9a8ef'], ['Vv', 0.3, '#e6d6fa'],
    ['A', 0.38, hex(COL.activations)], ['opt', 0.3, '#c9a8ef'], ['legend', 0.3, '#e8edf5'], ['note', 0.42, '#ffd9a8'], ['saved', 0.3, '#9be3c8'], ['title', 0.3, '#8793a6'],
    ['E0', 0.34, '#fff'], ['E0v', 0.3, '#fff'], ['E1', 0.34, '#fff'], ['E1v', 0.3, '#fff'], ['E2', 0.34, '#fff'], ['E2v', 0.3, '#fff'], ['E3', 0.34, '#fff'], ['E3v', 0.3, '#fff'],
  ];
  for (const [k, s, c] of labSpec) mkLab(k, s, c);
  const slabLab = Array.from({ length: MAXL }, () => mkLab('s' + Math.random(), 0.27, '#bff0de'));
  // roofline labels
  const opLab = Array.from({ length: 6 }, () => mkLab('p' + Math.random(), 0.38, '#e8edf5'));
  const barLab = Array.from({ length: 3 }, () => mkLab('b' + Math.random(), 0.4, '#fff'));
  const rLab = { ridge: mkLab('ridge', 0.38, '#ffffff'), xAxis: mkLab('xAxis', 0.36, '#aeb8cc'), yAxis: mkLab('yAxis', 0.36, '#aeb8cc'), chipnote: mkLab('chipnote', 0.3, '#aeb8cc'), assump: mkLab('assump', 0.3, '#8793a6'), step: mkLab('step', 0.4, '#e8edf5'), roofM: mkLab('roofM', 0.34, hex(BOUND_COL.memory)), roofC: mkLab('roofC', 0.34, hex(BOUND_COL.compute)) };
  const xTick = [-1, 0, 1, 2, 3].map((d) => mkLab('xt' + d, 0.32, '#aeb8cc')), yTick = [11, 12, 13, 14].map((d) => mkLab('yt' + d, 0.32, '#aeb8cc'));
  const zones = { m: mkLab('zm', 0.36, hex(BOUND_COL.memory)), c: mkLab('zc', 0.36, hex(BOUND_COL.compute)) };
  plotLabels();
  function plotLabels() {
    const xa = [0.1, 1, 10, 100, 1000];
    xTick.forEach((s, i) => { setT(s, String(xa[i])); s.position.set(MM.xOfAI(xa[i]), FLOOR + 0.12, zFront + 0.15); });
    const ya = ['100 GFLOP/s', '1 TFLOP/s', '10 TFLOP/s', '100 TFLOP/s'];
    yTick.forEach((s, i) => { setT(s, ya[i]); s.userData.wantW = true; });
  }

  // ---------------------------------------------------------------- state
  const S = {
    dirty: true, cfg: ctx.params(), L: 6, xs: new Float32Array(MAXL), dx: 1.47, n: 0, nu: MM.NU_REF, lay: null, tab: null, ppl: [], mem: null, extrasBytes: 0,
    // displayed (eased) geometry
    dsx: new Float32Array(MAXL), dsy: new Float32Array(MAXL), dsz: new Float32Array(MAXL), dpx: new Float32Array(MAXL),
    tsx: new Float32Array(MAXL), tsy: new Float32Array(MAXL), tsz: new Float32Array(MAXL), tpx: new Float32Array(MAXL),
    de: new Float32Array(NPAR), te: new Float32Array(NPAR),
    keep: new Float32Array(MAXL).fill(1), flash: new Float32Array(MAXL), grad: 1, opt: 1, upd: 0, wave: 0, u: 0,
    pillarX: new Float32Array(6), pillarH: new Float32Array(6), pillarC: Array.from({ length: 6 }, () => new THREE.Color(0xffffff)), pillarB: ['memory', 'memory', 'memory', 'memory', 'memory', 'over'],
    ridgeX: 3, barW: new Float32Array(3), vis: 1, hl: new Float32Array(MAXL), hlTouched: false, version: 0, readKey: '', shelf: [], blockPos: {},
  };
  const bw = { p: 0, playing: false, dur: 11, external: null, active: false };
  const extras = []; // [{id,label,color,bytes,gate(0..1 eased)}]
  const holds = new Set();
  const listeners = new Set();
  const col = new THREE.Color(), tmp = new THREE.Color(), V = new THREE.Vector3();
  const mobile = () => innerWidth <= 640;

  function refresh() {
    S.dirty = false; const cfg = S.cfg = ctx.params(); const L = S.L = cfg.depth + 2; const A = net();
    for (let l = 0; l < L; l++) S.xs[l] = A && A.layerCount === L ? A.getLayerAnchor(l).x : -0.15 + l * (7.35 / (L - 1));
    S.dx = (S.xs[L - 1] - S.xs[0]) / Math.max(1, L - 1);
    const ex = extras.map((e) => e.bytes);
    S.n = MM.pickScale(cfg, ex, S.dx, S.n); S.nu = MM.nuOf(S.n);
    const lay = S.lay = MM.layoutBlocks(cfg, ex, S.nu, S.dx); S.mem = lay.memory; S.ppl = MM.paramsPerLinear(cfg);
    for (let l = 0; l < MAXL; l++) { const s = lay.slabs[l]; S.tsx[l] = s ? Math.max(0.03, s.x) : 0; S.tsy[l] = s ? Math.max(0.03, s.y) : 0; S.tsz[l] = s ? Math.max(0.03, s.z) : 0; S.tpx[l] = l < L ? S.xs[l] : S.xs[L - 1]; }
    const e = lay.edges; const te = [e.weights, e.gradients, e.adamM, e.adamV, ...e.extras]; for (let i = 0; i < NPAR; i++) S.te[i] = te[i] ?? 0;
    S.tab = MM.opTable(cfg, 'physics'); S.ridgeX = MM.xOfAI(S.tab.ridge);
    // label text
    LS = mobile() ? 2.0 : 1;
    const m = S.mem, tot = m.total + ex.reduce((a, b) => a + b, 0), pct = (x) => Math.round(100 * x / tot) + '%', fb = model.fmtBytes;
    setT(labs.W, 'WEIGHTS'); setT(labs.Wv, `${fb(m.weights)} · ${pct(m.weights)}`);
    setT(labs.G, 'GRADIENTS'); setT(labs.Gv, `${fb(m.gradients)} · ${pct(m.gradients)}`);
    setT(labs.M, 'ADAM m'); setT(labs.Mv, `${fb(m.params * 4)} · ${pct(m.params * 4)}`);
    setT(labs.V, 'ADAM v'); setT(labs.Vv, `${fb(m.params * 4)} · ${pct(m.params * 4)}`);
    setT(labs.A, `ACTIVATIONS  ${fb(m.activations)} · ${pct(m.activations)}`);
    setT(labs.opt, `OPTIMIZER STATE = ${(m.optimizer / m.weights).toFixed(0)}× weights · ${fb(m.optimizer)}`);
    setT(labs.saved, 'saved for backward: each is kept until its gradient is computed');
    const bleg = Math.pow(2, Math.round(Math.log2(0.18 / S.nu)));
    S.legendBytes = bleg; setT(labs.legend, `= ${fb(bleg)}  (block volume = bytes)`);
    setT(labs.title, 'model estimate · not measured');
    extras.forEach((ex2, i) => { setT(labs['E' + i], ex2.label); setT(labs['E' + i + 'v'], `${fb(ex2.bytes)} · ${pct(ex2.bytes)}`); labs['E' + i].material.color.setStyle(hex(ex2.color)); });
    for (let l = 0; l < MAXL; l++) { if (l < L) setT(slabLab[l], fb(m.perLayerActivations[l])); }
    const T = S.tab; setT(rLab.ridge, `ridge ${T.ridge < 10 ? T.ridge.toFixed(1) : Math.round(T.ridge)} FLOP/B`);
    setT(rLab.chipnote, 'blocks each op streams (blue W, green A, orange G, purple Adam)'); setT(rLab.xAxis, 'arithmetic intensity, FLOP per byte (log)'); setT(rLab.yAxis, 'attainable FLOP/s (log)');
    setT(rLab.roofM, 'memory-bound'); setT(rLab.roofC, 'compute-bound'); setT(zones.m, 'MEMORY-BOUND side'); setT(zones.c, 'COMPUTE-BOUND side');
    setT(rLab.step, mobile() ? `step ${model.fmtTime(T.total)} (est.)` : `PINN step ${model.fmtTime(T.total)} · model estimate`); setT(rLab.assump, `overhead = ${T.launches} autograd nodes × ${MM.ASSUMPTIONS.launchUs} us (assumed)`);
    OPS_ALL.forEach((o, i) => { const op = i < 5 ? T.ops[i] : null; if (op) { setT(opLab[i], op.flops > 0 ? `${o.short} ${Math.max(0, Math.round(op.share * 100))}%` : `${o.short} (none)`); } else setT(opLab[5], `launch/graph overhead ${Math.round(T.overShare * 100)}%`); });
    setT(barLab[0], `memory-bound ${Math.round(T.memShare * 100)}%`); setT(barLab[1], `compute-bound ${Math.round(T.compShare * 100)}%`); setT(barLab[2], `overhead ${Math.round(T.overShare * 100)}%`);
    // roof curve + ridge + markers targets
    const cur = MM.roofCurve(cfg, NRF), pos = roofG.attributes.position.array, cl = roofG.attributes.color.array;
    for (let i = 0; i < NRF; i++) {
      const x = MM.xOfAI(cur[i].ai), y = MM.yOfPerf(cur[i].perf), mem = cur[i].ai < T.ridge; tmp.setHex(mem ? BOUND_COL.memory : BOUND_COL.compute);
      pos.set([x, y - 0.045, P.wallZ + 0.02, x, y + 0.045, P.wallZ + 0.02], i * 6); cl.set([tmp.r, tmp.g, tmp.b, tmp.r, tmp.g, tmp.b], i * 6);
    }
    roofG.attributes.position.needsUpdate = true; roofG.attributes.color.needsUpdate = true;
    const rp = ridgeG.attributes.position.array, peakY = MM.yOfPerf(S.cfg.peakTflops * 1e12), rx = S.ridgeX;
    rp.set([rx, PF + 0.02, zFront, rx, PF + 0.02, P.wallZ, rx, PF, P.wallZ, rx, peakY, P.wallZ]); ridgeG.attributes.position.needsUpdate = true;
    const mid = ((P.xL - 1.7) + rx) / 2; void mid;
    S.version++; for (const f of listeners) { try { f(api.getBreakdown()); } catch (e2) { console.error('memory listener', e2); } }
  }
  const OPS_ALL = [...MM.OPS, { id: 'over', short: 'overhead' }];

  // ---------------------------------------------------------------- helpers to write geometry
  const na = boxes.instanceMatrix.array, nc = () => boxes.instanceColor.array, ola = outline.geometry.attributes.position.array, olc = outline.geometry.attributes.color.array;
  let olN = 0;
  function putBox(i, cx, cy, cz, sx, sy, sz, c, bright) {
    const o = i * 16; na[o] = sx; na[o + 5] = sy; na[o + 10] = sz; na[o + 12] = cx; na[o + 13] = cy; na[o + 14] = cz;
    const k = nc(); k[i * 3] = c.r * bright; k[i * 3 + 1] = c.g * bright; k[i * 3 + 2] = c.b * bright;
  }
  function putOutline(cx, cy, cz, sx, sy, sz, c, a) {
    if (olN >= NB) return; const hx = sx / 2, hy = sy / 2, hz = sz / 2; let o = olN * 72; olN++;
    for (const [p, q] of EDGES) {
      for (const k of [p, q]) { ola[o] = cx + ((k & 1) ? hx : -hx); ola[o + 1] = cy + ((k & 2) ? hy : -hy); ola[o + 2] = cz + ((k & 4) ? hz : -hz); olc[o] = c.r * a; olc[o + 1] = c.g * a; olc[o + 2] = c.b * a; o += 3; }
    }
  }
  const linkA = links.geometry.attributes.position.array, linkC = links.geometry.attributes.color.array;
  function putLink(i, ax, ay, az, bx, by, bz, c, a) { const o = i * 6; linkA[o] = ax; linkA[o + 1] = ay; linkA[o + 2] = az; linkA[o + 3] = bx; linkA[o + 4] = by; linkA[o + 5] = bz; linkC[o] = linkC[o + 3] = c.r * a; linkC[o + 1] = linkC[o + 4] = c.g * a; linkC[o + 2] = linkC[o + 5] = c.b * a; }
  const cW = new THREE.Color(COL.weights), cG = new THREE.Color(COL.gradients), cA = new THREE.Color(COL.activations), cM = new THREE.Color(COL.optimizer), cV = new THREE.Color(0x7e57c2), cWhite = new THREE.Color(1, 1, 1);
  const parCol = [cW, cG, cM, cV];
  const place = (label, x, y, z, vis, w) => { label.position.set(x, y, z); label.visible = vis > 0.04; label.material.opacity = clamp(vis) * (w ?? 1); };

  // ---------------------------------------------------------------- per-frame
  function tick(dt, state) {
    if (disposed || lastStamp === state.time) return; lastStamp = state.time;
    const r = (id) => util.revealOf(state, id);
    const after = state.stageIndex > boundsIdx && holds.size === 0;
    S.vis += ((after ? 0 : 1) - S.vis) * (state.reducedMotion ? 1 : 1 - Math.exp(-dt * 5));
    // the roofline plot is a stage-7 view only: it fades after stage 7 even when other scenes hold() the memory blocks
    { const pa = state.stageIndex > boundsIdx ? 0 : 1; S.visPlot = ((S.visPlot ?? 1) + (pa - (S.visPlot ?? 1)) * (state.reducedMotion ? 1 : 1 - Math.exp(-dt * 5))); }
    const gW = r('memory.weights') * S.vis, gA = r('memory.activations') * S.vis, gG = r('memory.gradients') * S.vis, gO = r('memory.optimizer') * S.vis, gF = r('backward.flow') * S.vis, gB = r('bounds.overlay') * S.visPlot;
    const anyMem = Math.max(gW, gA, gG, gO), anyExtra = extras.some((e) => e.gate > 0.001);
    group.visible = anyMem > 0.001 || gB > 0.001 || anyExtra;
    if (!group.visible) { if (S.hlTouched) clearHl(); return; }
    if ((mobile() ? 2.0 : 1) !== LS) S.dirty = true;
    if (S.dirty) refresh();
    const L = S.L, ease = state.reducedMotion ? 1 : 1 - Math.exp(-dt * 9);

    // ---- backward timeline
    const inBwd = state.stageId === 'backward' || bw.external !== null; bw.active = inBwd;
    if (bw.external !== null) bw.p = bw.external;
    else if (inBwd && bw.playing && !state.reducedMotion) { bw.p = Math.min(1, bw.p + dt / bw.dur); if (bw.p >= 1) { bw.playing = false; syncUI(); } }
    if (inBwd && state.reducedMotion && bw.external === null) bw.p = 1;
    const bs = MM.backwardState(bw.p, L, S.ppl);
    const kc = inBwd ? 1 - Math.exp(-dt * 14) : 1 - Math.exp(-dt * 3);
    for (let l = 0; l < MAXL; l++) { const tk = inBwd && l < L ? bs.keep[l] : 1, tf = inBwd && l < L ? bs.flash[l] : 0; S.keep[l] += (tk - S.keep[l]) * kc; S.flash[l] += (tf - S.flash[l]) * (inBwd ? 0.5 : kc); }
    const tg = inBwd ? bs.gradFill : 1, to = inBwd ? bs.opt : 1, tu = inBwd ? bs.upd : 0;
    S.grad += (tg - S.grad) * kc; S.opt += (to - S.opt) * kc; S.upd += (tu - S.upd) * (inBwd ? 0.5 : 1); S.u = bs.u;
    S.wave = inBwd ? bs.wave : 0;
    // ---- ease geometry targets
    let moving = false;
    for (let l = 0; l < MAXL; l++) {
      for (const [d, t] of [[S.dsx, S.tsx], [S.dsy, S.tsy], [S.dsz, S.tsz], [S.dpx, S.tpx]]) { const v = d[l] + (t[l] - d[l]) * ease; d[l] = Math.abs(v - t[l]) < 1e-4 ? t[l] : v; if (d[l] !== t[l]) moving = true; }
    }
    for (let i = 0; i < NPAR; i++) { const v = S.de[i] + (S.te[i] - S.de[i]) * ease; S.de[i] = Math.abs(v - S.te[i]) < 1e-4 ? S.te[i] : v; }
    for (const e of extras) { const t = e.removing ? 0 : 1; e.gate += (t - e.gate) * ease; if (e.removing && e.gate < 0.01) e.dead = true; }
    for (let i = extras.length - 1; i >= 0; i--) if (extras[i].dead) { extras.splice(i, 1); S.dirty = true; }
    void moving;
    olN = 0;

    // ---- activation slabs
    let maxTop = FLOOR, xmin = 1e9, xmax = -1e9;
    for (let l = 0; l < MAXL; l++) {
      const i = l, kr = l < L ? ss(l / L * 0.6, l / L * 0.6 + 0.4, gA) : 0, sxx = S.dsx[l] * kr, kp = S.keep[l];
      const shr = 0.1 + 0.9 * kp, sy = S.dsy[l] * kr * shr, sz = S.dsz[l] * kr * shr, cx = S.dpx[l];
      if (kr < 0.002 || sxx < 1e-3) { putBox(i, 0, -50, 0, 0, 0, 0, cA, 0); continue; }
      const fl = S.flash[l], bright = 0.45 + 0.55 * kp + 0.9 * fl;
      col.copy(cA).lerp(cWhite, clamp(0.55 * fl)); putBox(i, cx, FLOOR + sy / 2, 0, sxx, sy, sz, col, bright);
      const fy = S.dsy[l] * kr, fz = S.dsz[l] * kr;
      putOutline(cx, FLOOR + fy / 2, 0, sxx, fy, fz, cA, (0.3 + 0.5 * (1 - kp)) * kr);
      maxTop = Math.max(maxTop, FLOOR + fy); xmin = Math.min(xmin, cx); xmax = Math.max(xmax, cx);
      // stem from network layer down to the slab (activation is "attached" to its layer)
      putLink(l, cx, -1.05, 0, cx, FLOOR + sy, 0, cA, 0.55 * kr * (0.35 + 0.65 * kp));
    }
    for (let l = L; l < MAXL; l++) putLink(l, 0, 0, 0, 0, 0, 0, cA, 0);
    // ---- parameter-type blocks on the shelf
    let sx0 = SHELF_X0; const gates = [gW, gG, gO, gO, ...extras.map((e) => Math.min(e.gate, 1) * S.vis), 0, 0, 0, 0].slice(0, NPAR); const sPos = S.shelf; sPos.length = 0;
    const shown = (i) => (i < 4 || S.de[i] > 0.02);
    const hub = { x: 0, y: 0, z: 0 };
    for (let i = 0; i < NPAR; i++) {
      const e = S.de[i]; if (!shown(i)) { sPos.push(null); putBox(I_PAR + i, 0, -50, 0, 0, 0, 0, cW, 0); continue; }
      const cx = sx0 + e / 2, cz = SHELF_Z0 + e / 2; sx0 += e + MM.LIMITS.gap; sPos.push({ cx, cz, e });
      const g = ss(0, 1, gates[i]), base = i < 4 ? parCol[i] : (tmp.setHex(extras[i - 4]?.color ?? 0xffffff), tmp);
      let h = e * g, fill = 1, w = e * g;
      if (i === 1) fill = S.grad; if (i === 2 || i === 3) { const o = ss(0, 1, S.opt); w = e * g * o; h = w; }
      if (i === 1) h = e * g * clamp(fill);
      let bright = 1; col.copy(base);
      if (i === 0) { const pulse = bs.upd > 0 && bs.upd < 1 || S.upd > 0.02 ? Math.sin(Math.min(1, S.upd) * Math.PI * 6) * 0.5 + 0.5 : 0; col.lerp(cWhite, 0.55 * pulse * clamp(S.upd * 4)); bright = 1 + 0.5 * pulse; if (S.upd > 0.02) { w *= 1 + 0.05 * pulse; h *= 1 + 0.05 * pulse; } }
      if (h * w < 1e-6) putBox(I_PAR + i, 0, -50, 0, 0, 0, 0, base, 0); else putBox(I_PAR + i, cx, FLOOR + h / 2, cz, w, h, w, col, bright);
      putOutline(cx, FLOOR + e * g / 2, cz, e * g, e * g, e * g, base, 0.75 * g);
      S.blockPos[['weights', 'gradients', 'adamM', 'adamV', 'e0', 'e1', 'e2', 'e3'][i]] = { x: cx, y: FLOOR + e / 2, z: cz, size: e };
      if (i === 0) { hub.x = cx; hub.y = FLOOR + e / 2; hub.z = cz; }
    }
    // legend cube
    { const e = MM.cubeEdge(S.legendBytes, S.nu), g = ss(0, 1, Math.max(gW, gA)); const lz = SHELF_Z0 + 2.7; putBox(I_LEG, LEG_X, FLOOR + e * g / 2, lz, e * g, e * g, e * g, cWhite, 0.85); putOutline(LEG_X, FLOOR + e * g / 2, lz, e * g, e * g, e * g, cWhite, 0.8 * g); S.blockPos.legend = { x: LEG_X, y: FLOOR + e / 2, z: lz, size: e }; }
    boxes.instanceMatrix.needsUpdate = true; boxes.instanceColor.needsUpdate = true;
    outline.geometry.setDrawRange(0, olN * 24); for (let i = olN * 72; i < NB * 72; i++) olc[i] = 0;
    outline.geometry.attributes.position.needsUpdate = true; outline.geometry.attributes.color.needsUpdate = true;
    // beams: slab -> gradient block when the gradient of the following linear layer is produced
    const Gp = sPos[1]; let li = MAXL;
    if (Gp && inBwd) {
      for (let l = 0; l < L - 1 && li < MAXL * 2; l++) { const f = S.flash[l] * gG * 1.4; putLink(li++, S.dpx[l], FLOOR + S.dsy[l] * (0.1 + 0.9 * S.keep[l]), 0, Gp.cx, FLOOR + Gp.e * 0.9, Gp.cz, cG, f); }
    }
    for (; li < MAXL * 2; li++) putLink(li, 0, 0, 0, 0, 0, 0, cG, 0);
    links.geometry.setDrawRange(0, MAXL * 4); links.geometry.attributes.position.needsUpdate = true; links.geometry.attributes.color.needsUpdate = true;
    links.visible = outline.visible = boxes.visible = true;
    // ---- optimizer particles (G -> m,v while m,v appear; G,m,v -> W during update)
    { const pa = parts.instanceMatrix.array, pcol = parts.instanceColor.array; const gP = sPos[1], mP = sPos[2], vP = sPos[3], wP = sPos[0];
      const phA = inBwd ? bs.opt : 0, phU = S.upd;
      for (let i = 0; i < NP; i++) {
        let s = 0, x = 0, y = 0, z = 0, colr = cG, size = 0; const o = i * 16;
        if (gP && mP && vP && wP) {
          const f = i / NP;
          if (phU > 0.001 && phU < 0.999) {
            const src = [gP, mP, vP][i % 3], t = clamp(phU * 2.4 - f * 0.9, 0, 1); if (t > 0 && t < 1) {
              s = t; x = src.cx + (wP.cx - src.cx) * s; z = src.cz + (wP.cz - src.cz) * s; y = FLOOR + 0.3 + src.e * 0.5 + Math.sin(Math.PI * s) * 1.0; colr = i % 3 === 0 ? cG : cM; size = 0.075 * Math.sin(Math.PI * s) ** 0.5;
            }
          } else if (phA > 0.02 && phA < 0.98 && bw.p < 0.78) {
            const dst = i % 2 ? vP : mP, t = clamp(phA * 1.8 - f * 0.8, 0, 1); if (t > 0 && t < 1) {
              s = t; x = gP.cx + (dst.cx - gP.cx) * s; z = gP.cz + (dst.cz - gP.cz) * s; y = FLOOR + 0.3 + gP.e * 0.5 + Math.sin(Math.PI * s) * 0.7; colr = cM; size = 0.07 * Math.sin(Math.PI * s) ** 0.5;
            }
          }
        }
        pa[o] = size; pa[o + 5] = size; pa[o + 10] = size; pa[o + 12] = x; pa[o + 13] = y; pa[o + 14] = z; pcol[i * 3] = colr.r; pcol[i * 3 + 1] = colr.g; pcol[i * 3 + 2] = colr.b;
      }
      parts.instanceMatrix.needsUpdate = true; parts.instanceColor.needsUpdate = true; parts.visible = inBwd || S.upd > 0.001;
    }
    // ---- wavefront + network highlight
    { const wv = S.wave * gF, uu = clamp(S.u, 0, L - 1), l0 = Math.floor(uu), l1 = Math.min(L - 1, l0 + 1), x = S.xs[l0] + (S.xs[l1] - S.xs[l0]) * (uu - l0);
      const h = 1.6 - FLOOR + 0.4;
      waves.forEach((m, k) => { m.visible = wv > 0.01; m.position.set(x + k * 0.28, (FLOOR + 1.6) / 2 - 0.1, 0); m.scale.set(8.0, h, 1); m.material.opacity = m.userData.base * wv; });
      waveEdge.material.opacity = 0.9 * wv;
      const A = net();
      if (A && (inBwd && wv > 0.01 || S.hlTouched)) {
        if (inBwd && wv > 0.01) { for (let l = 0; l < L; l++) { const a = clamp(0.85 * S.flash[l] + 0.25 * (1 - S.keep[l])); if (Math.abs(a - S.hl[l]) > 0.02) { S.hl[l] = a; A.setActivationHighlight(l, a, COL.gradients); } } S.hlTouched = true; }
        else clearHl();
      }
      const wl = labs.note; const lay = Math.round(uu), lk = inBwd && bw.p > 0 && bw.p < BWD_END ? Math.min(L - 1, Math.max(0, lay)) : -1;
      const mo = mobile();
      if (lk >= 1) setT(wl, mo ? `dW${sub(lk)} = a${sub(lk - 1)}ᵀ·δ${sub(lk)} → a${sub(lk - 1)} freed` : `dW${sub(lk)} = a${sub(lk - 1)}ᵀ · δ${sub(lk)}   (needs the SAVED a${sub(lk - 1)}) → then a${sub(lk - 1)} is freed`);
      else if (lk === 0) setT(wl, mo ? 'input a₀ used, freed' : 'layer 1 done: saved input a₀ used, then freed');
      else if (inBwd) setT(wl, bw.p >= BWD_END ? (mo ? 'Adam m, v → weights updated' : 'gradients complete → Adam keeps m, v (2× weights) → weights updated') : (mo ? 'press Replay' : 'press Replay to watch autograd consume the saved activations'));
      place(wl, 3.6, FLOOR + 0.2, SHELF_Z0 + 3.9, inBwd ? gF : 0);
    }
    // ---- memory labels
    { const lv = Math.max(0, 1 - gB * 1.0);   // dim the small per-layer labels once the roofline is on screen
      const sp = (i) => sPos[i];
      const hm = mobile() && state.stageIndex > boundsIdx ? 0 : 1;   // phones: after stage 7 only the held extra block (autograd graph) keeps labels, so stages 11-14 stay readable
      const lab2 = (nameL, valL, i, g) => { const p = sp(i); if (!p) { nameL.visible = valL.visible = false; return; } const top = FLOOR + p.e + 0.3; place(nameL, p.cx, top + 0.4 * LS, p.cz, g); place(valL, p.cx, top, p.cz, g); };
      lab2(labs.W, labs.Wv, 0, gW * hm); lab2(labs.G, labs.Gv, 1, gG * hm); lab2(labs.M, labs.Mv, 2, gO * hm); lab2(labs.V, labs.Vv, 3, gO * hm);
      extras.forEach((e, i) => lab2(labs['E' + i], labs['E' + i + 'v'], 4 + i, Math.min(e.gate, 1) * S.vis)); for (let i = extras.length; i < 4; i++) { labs['E' + i].visible = labs['E' + i + 'v'].visible = false; }
      const cxm = (xmin + xmax) / 2 || 3.5, hdr = FLOOR + maxTop - FLOOR + 0.5 * LS;
      if (mobile()) place(labs.A, 2.3 + labs.A.scale.x / 2, FLOOR + 0.15, SHELF_Z0 - 0.75, gA * hm); else place(labs.A, cxm, hdr, 0, gA); place(labs.saved, cxm, hdr + 0.42 * LS, 0, mobile() ? 0 : gA * (inBwd ? 0.0 : 1) * lv);
      const p2 = sp(2), p3 = sp(3); if (p2 && p3) place(labs.opt, (p2.cx + p3.cx) / 2, FLOOR + Math.max(p2.e, p3.e) + 1.15 * LS + 0.3, (p2.cz + p3.cz) / 2, gO * S.opt * hm);
      else labs.opt.visible = false;
      const zr = SHELF_Z0 + 2.7, gl = Math.max(gW, gA) * lv * hm;
      place(labs.legend, LEG_X + 0.5 + labs.legend.scale.x / 2, FLOOR + 0.15, zr, gl); place(labs.title, mobile() ? 0 : LEG_X + 0.5 + labs.legend.scale.x + 0.3 + labs.title.scale.x / 2, FLOOR + 0.15, zr, mobile() ? 0 : gl);
      for (let l = 0; l < MAXL; l++) { const s = slabLab[l]; const kr = l < L ? ss(l / L * 0.6, l / L * 0.6 + 0.4, gA) : 0; const showIt = !mobile() && (L <= 8 || l % 2 === 0); place(s, S.dpx[l], FLOOR + 0.12, SHELF_Z0 - 0.7, showIt ? kr * lv * (inBwd ? 0.35 + 0.65 * S.keep[l] : 1) : 0); }
    }
    // ---- roofline plot
    plot.visible = gB > 0.001;
    if (plot.visible) roofline(dt, ease, gB, state);
    // ---- HUD
    hudUpdate(state);
  }
  const BWD_END = 0.78;
  const sub = (n) => String.fromCharCode(0x2080 + (n % 10)) ;
  function clearHl() { const A = net(); if (A) for (let l = 0; l < MAXL; l++) if (S.hl[l] !== 0) { S.hl[l] = 0; A.setActivationHighlight(l, 0); } S.hlTouched = false; }

  function roofline(dt, ease, gB, state) {
    const T = S.tab, cfg = S.cfg;
    plot.traverse((o) => { const m = o.material; if (m && m.userData.base !== undefined && o !== pillars) m.opacity = m.userData.base * gB; });
    pillars.material.opacity = 0.92 * gB;
    // zones
    const xl = P.xR, xr = S.ridgeX;
    zoneM.position.set((P.xL + xr) / 2, PF + 0.003, zMid); zoneM.scale.set(Math.max(0.01, xr - P.xL), zLen, 1);
    zoneC.position.set((xr + P.xR) / 2, PF + 0.003, zMid); zoneC.scale.set(Math.max(0.01, P.xR - xr), zLen, 1);
    zoneO.position.set((xl + P.xO + 0.9) / 2, PF + 0.003, zMid); zoneO.scale.set(P.xO + 0.9 - xl, zLen, 1);
    // pillars: ops 0..4 then overhead (5), bar segments 6..8
    const pa = pillars.instanceMatrix.array, pc = pillars.instanceColor.array, ma = marks.instanceMatrix.array, mc = marks.instanceColor.array, dp = dropG.attributes.position.array;
    for (let i = 0; i < 6; i++) {
      const op = i < 5 ? T.ops[i] : null, active = i < 5 ? op.flops > 0 : true;
      const tx = i < 5 ? MM.xOfAI(op.intensity) : P.xO, share = i < 5 ? op.share : T.overShare, th = active ? Math.max(0.05, share * P.hMax) : 0.0;
      const bound = i < 5 ? op.bound : 'over'; S.pillarB[i] = bound;
      const e2 = S.pInit ? ease : 1; S.pillarX[i] += (tx - S.pillarX[i]) * e2; S.pillarH[i] += (th - S.pillarH[i]) * e2;
      tmp.setHex(bound === 'over' ? OVER_COL : BOUND_COL[bound]); S.pillarC[i].lerp(tmp, ease * 0.9 + 0.05);
      const z = MM.laneZ(i), h = S.pillarH[i], o = i * 16, kk = 0.8 + 0.2 * gB;
      pa[o] = 0.55 * kk; pa[o + 5] = Math.max(h, 1e-4); pa[o + 10] = 0.6 * kk; pa[o + 12] = S.pillarX[i]; pa[o + 13] = PF + h / 2; pa[o + 14] = z;
      pc[i * 3] = S.pillarC[i].r; pc[i * 3 + 1] = S.pillarC[i].g; pc[i * 3 + 2] = S.pillarC[i].b;
      // labels above pillars
      const lb = opLab[i]; place(lb, clamp(S.pillarX[i], P.xL + 0.3, P.xO), PF + h + 0.32 * LS, z, (active ? gB : gB * 0.35) * (mobile() && share < 0.05 ? 0 : 1));
      if (i < 5) {
        const y = MM.yOfPerf(op.perf), mo = i * 16; const rr = active ? 0.17 : 0;
        ma[mo] = rr; ma[mo + 5] = rr; ma[mo + 10] = rr; ma[mo + 12] = S.pillarX[i]; ma[mo + 13] = y; ma[mo + 14] = P.wallZ + 0.05; mc[i * 3] = S.pillarC[i].r; mc[i * 3 + 1] = S.pillarC[i].g; mc[i * 3 + 2] = S.pillarC[i].b;
        dp.set(active ? [S.pillarX[i], y, P.wallZ + 0.05, S.pillarX[i], PF, P.wallZ + 0.05] : [0, 0, 0, 0, 0, 0], i * 6);
      }
    }
    // stacked step-time bar (normalised to 100% of the step) at the front of the plot
    const W = P.xR - P.xL, sh = [T.memShare, T.compShare, T.overShare], cols = [BOUND_COL.memory, BOUND_COL.compute, OVER_COL]; let acc = 0;
    for (let k = 0; k < 3; k++) {
      const w = Math.max(1e-4, sh[k] * W), o = (6 + k) * 16, cx = P.xL + acc + w / 2; acc += sh[k] * W; S.barW[k] += (w - S.barW[k]) * ease;
      pa[o] = S.barW[k] * (0.92 + 0.08 * gB); pa[o + 5] = 0.42; pa[o + 10] = 0.5; pa[o + 12] = cx; pa[o + 13] = PF + 0.21; pa[o + 14] = P.bar.z;
      tmp.setHex(cols[k]); pc[(6 + k) * 3] = tmp.r; pc[(6 + k) * 3 + 1] = tmp.g; pc[(6 + k) * 3 + 2] = tmp.b;
      place(barLab[k], cx, PF + 0.25, P.bar.z + 0.3, sh[k] > 0.12 ? gB : 0);
    }
    S.pInit = true;
    pillars.instanceMatrix.needsUpdate = true; pillars.instanceColor.needsUpdate = true; marks.instanceMatrix.needsUpdate = true; marks.instanceColor.needsUpdate = true; dropG.attributes.position.needsUpdate = true;
    // static-ish labels
    const yTop = PF + P.wallH;
    place(rLab.chipnote, P.xL - 1.9 + rLab.chipnote.scale.x / 2, PF + 0.12, MM.laneZ(5) - 0.85, mobile() ? 0 : gB);
    place(rLab.ridge, S.ridgeX, yTop + 0.5, P.wallZ, gB);
    place(rLab.xAxis, (P.xL + P.xR) / 2, PF + 0.1, zFront + 0.95, mobile() ? 0 : gB); place(rLab.yAxis, P.xL - 0.8 + 0.2, yTop + 0.5, P.wallZ, gB);
    const mo = mobile(), gd = mo ? 0 : gB;
    place(rLab.step, (P.xL + P.xR) / 2, PF + 0.12, P.bar.z + 1.55, gB); place(rLab.assump, (P.xL + P.xR) / 2, PF + 0.12, P.bar.z + 2.2, gd);
    place(rLab.roofM, Math.max(P.xL + 0.9, MM.xOfAI(0.5)), MM.yOfPerf(cfg.bandwidthGBs * 1e9 * 0.5) + 0.45, P.wallZ, gB); place(rLab.roofC, Math.min(P.xR - 0.9, (S.ridgeX + P.xR) / 2 + 0.2), MM.yOfPerf(cfg.peakTflops * 1e12) + 0.45, P.wallZ, gB);
    place(zones.m, (P.xL + S.ridgeX) / 2, PF + 0.05, zBack - 0.15, S.ridgeX - P.xL > 3 ? gB : 0); place(zones.c, (S.ridgeX + P.xR) / 2, PF + 0.05, zBack - 0.15, P.xR - S.ridgeX > 3 ? gB : 0);
    yTick.forEach((s, i) => { place(s, P.xL - 0.3 - s.scale.x / 2, MM.yOfPerf(Math.pow(10, 11 + i)), P.wallZ, mobile() ? 0 : gB); });
    xTick.forEach((s) => { s.visible = gB > 0.04 && !mobile(); s.material.opacity = gB; });
    if (mobile()) { rLab.roofM.visible = rLab.roofC.visible = zones.m.visible = zones.c.visible = rLab.yAxis.visible = false; }
    void dt; void state;
  }

  // ---------------------------------------------------------------- HUD readouts
  function hudUpdate(state) {
    const id = state.stageId; if (id !== 'memory-fwd' && id !== 'backward' && id !== 'bounds') return;
    const T = S.tab, m = S.mem, fb = model.fmtBytes, key = id + S.version + (id === 'backward' ? Math.round(bw.p * 50) : '');
    if (key === S.readKey) return; S.readKey = key;
    if (id === 'bounds') {
      hud.readout('mem', 'Memory-bound', `${Math.round(T.memShare * 100)}% of step`, 'model estimate');
      hud.readout('comp', 'Compute-bound', `${Math.round(T.compShare * 100)}% of step`);
      hud.readout('over', 'Launch/graph overhead', `${Math.round(T.overShare * 100)}% of step`);
      hud.readout('step', 'Step time (PINN)', model.fmtTime(T.total));
      hud.readout('ridge', 'Ridge point', `${T.ridge < 10 ? T.ridge.toFixed(1) : Math.round(T.ridge)} FLOP/B`);
      hud.readout('mem2', 'Memory-bound, GPU-busy only', `${Math.round(T.memShareBusy * 100)}%`);
    } else {
      const tot = m.total; const pc = (x) => Math.round(100 * x / tot) + '%';
      hud.readout('w', 'Weights', `${fb(m.weights)} (${pc(m.weights)})`, 'model estimate'); hud.readout('a', 'Activations', `${fb(m.activations)} (${pc(m.activations)})`);
      if (id === 'backward') { hud.readout('g', 'Gradients', `${fb(m.gradients)} (${pc(m.gradients)})`); hud.readout('o', 'Adam m + v', `${fb(m.optimizer)} (${pc(m.optimizer)})`); }
      hud.readout('t', id === 'backward' ? 'Peak total' : 'Weights + activations', fb(id === 'backward' ? tot : m.weights + m.activations));
    }
  }

  // ---------------------------------------------------------------- UI (backward controls)
  const ui = document.createElement('div'); ui.id = 'mem-controls'; ui.className = 'card'; ui.setAttribute('role', 'group'); ui.setAttribute('aria-label', 'Backward pass controls');
  Object.assign(ui.style, { display: 'none', top: '86px', left: '12px', padding: '6px 8px', gap: '6px', alignItems: 'center', zIndex: 5, flexWrap: 'wrap', maxWidth: 'min(420px, calc(100vw - 24px))' });
  const mkBtn = (txt, label) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = txt; b.setAttribute('aria-label', label); Object.assign(b.style, { background: '#121a2b', color: '#e8edf5', border: '1px solid #26314a', borderRadius: '8px', padding: '5px 10px', cursor: 'pointer', font: 'inherit' }); return b; };
  const bReplay = mkBtn('↻ Replay', 'Replay backward pass'), bPlay = mkBtn('❚❚ Pause', 'Pause or play backward pass');
  const scrub = document.createElement('input'); scrub.type = 'range'; scrub.min = 0; scrub.max = 1000; scrub.value = 0; scrub.setAttribute('aria-label', 'Backward pass progress'); Object.assign(scrub.style, { width: '120px', accentColor: '#f08a24' });
  const uiNote = document.createElement('span'); Object.assign(uiNote.style, { color: '#8793a6', fontSize: '12px' });
  ui.append(bReplay, bPlay, scrub, uiNote); document.body.appendChild(ui);
  bReplay.onclick = () => { bw.external = null; bw.p = 0; bw.playing = true; syncUI(); };
  bPlay.onclick = () => { if (bw.p >= 1) bw.p = 0; bw.playing = !bw.playing; bw.external = null; syncUI(); };
  scrub.oninput = () => { bw.external = null; bw.playing = false; bw.p = scrub.value / 1000; syncUI(); };
  function syncUI() { bPlay.textContent = bw.playing ? '❚❚ Pause' : '▶ Play'; if (document.activeElement !== scrub) scrub.value = Math.round(bw.p * 1000); uiNote.textContent = bw.p >= 1 ? 'done' : bw.p < BWD_END ? 'backward' : bw.p < 0.9 ? 'Adam state' : 'update'; }
  const uiTimer = setInterval(() => { if (ui.style.display !== 'none') syncUI(); }, 200);
  const showUI = (on) => { ui.style.display = on ? 'flex' : 'none'; };

  // ---------------------------------------------------------------- public API
  const lazyNetSub = { off: null };
  const api = {
    /** Block layout version; increments on every layout refresh. */
    get version() { return S.version; },
    /** Memory breakdown + roofline step breakdown (all MODEL ESTIMATE). */
    getBreakdown() {
      const m = S.mem ?? model.memoryBreakdown(S.cfg), T = S.tab ?? MM.opTable(S.cfg), exB = extras.reduce((a, e) => a + e.bytes, 0);
      const total = m.total + exB;
      return { cfg: S.cfg, layers: S.L, bytes: { weights: m.weights, activations: m.activations, gradients: m.gradients, optimizer: m.optimizer, adamM: m.params * 4, adamV: m.params * 4, extras: exB, total, perLayerActivations: m.perLayerActivations.slice() },
        share: { weights: m.weights / total, activations: m.activations / total, gradients: m.gradients / total, optimizer: m.optimizer / total, extras: exB / total },
        scale: { step: S.n, voltsPerByte: S.nu, legendBytes: S.legendBytes },
        step: { time: T.total, memShare: T.memShare, compShare: T.compShare, overShare: T.overShare, memShareBusy: T.memShareBusy, ridge: T.ridge, launches: T.launches },
        ops: T.ops.map((o) => ({ id: o.id, name: o.name, intensity: o.intensity, bound: o.bound, share: o.share, time: o.time, flops: o.flops, bytes: o.bytes })), assumptions: MM.ASSUMPTIONS };
    },
    /** Roofline ops table for any step variant ('physics' | 'supervised' | 'sage'); pure model estimate. */
    getOps(variant = 'physics', cfg) { return MM.opTable(cfg ?? ctx.params(), variant); },
    /** Live scene state of the roofline pillars: what is actually drawn (id, bound, colour, x, height). */
    getPillars() { return MM.OPS.concat([{ id: 'over' }]).map((o, i) => ({ id: o.id, bound: S.pillarB[i], color: S.pillarC[i].getHex(), x: S.pillarX[i], height: S.pillarH[i], target: i < 5 ? S.tab.ops[i].bound : 'over' })); },
    /** World anchor (fresh Vector3) of a block's top-centre: 'weights'|'gradients'|'adamM'|'adamV'|'e0'..'e3'|'legend'|'act' (layer l) . */
    getBlockAnchor(id, l = 0) {
      if (id === 'act') { const x = S.dpx[l] ?? 0; return new THREE.Vector3(x, FLOOR + (S.dsy[l] ?? 0), 0); }
      const b = S.blockPos[id]; return b ? new THREE.Vector3(b.x, b.y + b.size / 2, b.z) : null;
    },
    /** Scripted backward timeline: p in [0,1] (pauses autoplay); null returns control to the stage's own animation. */
    setBackwardProgress(p) { bw.external = p === null ? null : clamp(p); if (p !== null) { bw.p = bw.external; bw.playing = false; } syncUI(); },
    get backwardProgress() { return bw.p; },
    replayBackward() { bw.external = null; bw.p = 0; bw.playing = true; syncUI(); },
    /** Add/update/remove a block on the parameter shelf (max 4), e.g. {id:'autograd', label:'AUTOGRAD GRAPH', bytes, color:0xe0408a}. null removes. Volume = bytes like all blocks. */
    setExtraBlock(id, spec) {
      const i = extras.findIndex((e) => e.id === id);
      if (spec === null) { if (i >= 0) extras[i].removing = true; return; }
      if (i >= 0) { Object.assign(extras[i], spec, { removing: false }); } else if (extras.length < 4) extras.push({ id, label: spec.label ?? id, color: spec.color ?? 0xe0408a, bytes: spec.bytes, gate: 0 });
      S.dirty = true;
    },
    /** Keep the memory blocks on screen after stage 7 (normally they fade out when the story moves on). hold(owner, bool). */
    hold(owner, on = true) { if (on) holds.add(owner); else holds.delete(owner); },
    onLayout(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    /** Plot constants and mappings (for scenes drawing on the roofline). */
    plot: { ...P, xOfAI: MM.xOfAI, yOfPerf: MM.yOfPerf, laneZ: MM.laneZ }, FLOOR_Y: FLOOR,
  };

  // ---------------------------------------------------------------- knobs
  const ids = ['batch', 'width', 'depth', 'precision', 'bandwidth', 'peakTflops'];
  const offs = ids.map((id) => knobs.on(id, () => { S.dirty = true; }));
  let netOff = null; const subNet = () => { if (netOff || !net()) return; netOff = net().onRebuild(() => { S.dirty = true; }); };
  void lazyNetSub;

  // ---------------------------------------------------------------- stages
  const cam = (box, o) => { const opt = () => o ?? (mobile() ? (box === bndBox ? MOBB : MOB) : { az: -18, el: box === bndBox ? 28 : 18 }); return { get pos() { return frameBox(THREE, ctx.camera, box(), opt()).pos; }, get target() { return frameBox(THREE, ctx.camera, box(), opt()).target; } }; };
  const memBox = () => (mobile() ? [-2.4, 9.2, FLOOR - 0.3, FLOOR + 2.8, -2.2, SHELF_Z0 + 4.4] : [-2.2, 8.8, FLOOR - 0.5, 1.9, -2.2, SHELF_Z0 + 4.2]);
  const MOB = { az: -32, el: 42 }, MOBB = { az: -75, el: 55 };
  const bndBox = () => (mobile() ? [P.xL - 2.4, P.xO + 1.2, PF - 0.4, PF + P.wallH + 0.9, P.wallZ - 0.3, P.bar.z + 2.8] : [P.xL - 2.8, P.xO + 1.6, PF - 0.4, FLOOR + 2.2, P.wallZ - 0.3, SHELF_Z0 + 4.2]);   // desktop: plot AND the memory blocks above it stay in view
  const handler = (extra) => ({ enter() { subNet(); S.readKey = ''; S.dirty = true; extra?.enter?.(); }, exit() { extra?.exit?.(); }, update: tick, ...(extra?.rest ?? {}) });
  const stages = {
    'memory-fwd': handler({ rest: { camera: cam(memBox) } }),
    backward: handler({ enter() { bw.external = null; bw.p = ctx.state.reducedMotion ? 1 : 0; bw.playing = !ctx.state.reducedMotion; showUI(true); syncUI(); }, exit() { showUI(false); bw.playing = false; },
      rest: { camera: cam(memBox), caption: 'Autograd walks output to input. Each activation was SAVED in the forward pass because its gradient needs it (dW = aᵀ·δ); once used it is freed (slab shrinks) while the GRADIENT block fills. Then Adam adds m and v (2× weights) and updates the weights.' } }),
    bounds: handler({ rest: { camera: cam(bndBox) } }),
  };
  return {
    group, api, stages,
    dispose() {
      disposed = true; clearInterval(uiTimer); offs.forEach((f) => f()); netOff?.(); listeners.clear(); ui.remove();
      util.disposeObject(group); [boxGeo, sphGeo, planeGeo, edgeGeo, gridG, plotLinesG, roofG, ridgeG, dropG].forEach((g) => g.dispose());
      [boxes, parts, pillars, marks, chips].forEach((m) => m.dispose());
    },
  };
}
