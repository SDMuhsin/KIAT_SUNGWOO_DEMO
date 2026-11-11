// main.js - the single scene: input geometry -> sampled points -> network -> illustrative output, with the two autograd graphs below.
// One training step loops (Play/Pause/Replay/scrub). One SAGE toggle morphs the physics graph into the thin v lane (3.6 s, reversible, scrubbable).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { BG, COLORS } from './theme.js';
import { clamp, lerp, easeInOutCubic, makeLabel, setLabel, disposeObject } from './util.js';
import { createKnobs } from './knobs.js';
import { paramsFromKnobs, sageEndpoints, fmtBytes, fmtTime, FACTS } from './model.js';
import { makePack, PACK } from './geo.js';
import { createCloud, drawCount } from './cloud.js';
import { createNet, NET } from './net.js';
import { createGraph, GR } from './graph.js';
import { timeline, prog, phaseAt } from './anim.js';

const $ = (id) => document.getElementById(id);
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const mobile = () => innerWidth < 700;

// ---------- renderer / camera ----------
const canvas = $('stage');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setClearColor(BG, 1);
const scene = new THREE.Scene(), world = new THREE.Group(); scene.add(world); // world rotates 90 degrees on portrait phones (flow runs top to bottom)
const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 400);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = !reduced; controls.dampingFactor = 0.08; controls.minDistance = 6; controls.maxDistance = 120; controls.maxPolarAngle = Math.PI * 0.62;
const BOX = { min: new THREE.Vector3(-20.6, -13.6, -3), max: new THREE.Vector3(15.4, 3.6, 3) };
const CENTER = BOX.min.clone().add(BOX.max).multiplyScalar(0.5);
let userMoved = false, dirty = true, portrait = false;
const labelSprites = [];

// ---------- world ----------
const IN_X = -16.2, OUT_X = 11.6, IN_S = 1.5, OUT_S = 1.3;
const input = makePack(THREE); input.group.position.set(IN_X, 0, 0); input.group.scale.setScalar(IN_S); world.add(input.group);
const output = makePack(THREE, { field: true }); output.group.position.set(OUT_X, 0, 0); output.group.scale.setScalar(OUT_S); world.add(output.group);
const cloud = createCloud(THREE); // destination (network input) is set per layout in the cloud group's local space
cloud.group.position.set(IN_X, 0, 0); cloud.group.scale.setScalar(IN_S);
world.add(cloud.group);
const net = createNet(THREE); world.add(net.group);

const LOSS = { data: new THREE.Vector3(6.4, -3.0, 0), phys: new THREE.Vector3(6.4, -4.3, 0) };
const Y = { dataF: -4.6, dataB: -5.9, physF: -8.7, physB: -11.9 };
const lane = { a: LOSS.phys.clone(), b: new THREE.Vector3(GR.x1 + 0.2, Y.dataB, 0), c: new THREE.Vector3(5.8, -6.4, 0.7) };
const dataGraph = createGraph(THREE, 'supervised', Y.dataF, Y.dataB, lane); world.add(dataGraph.group);
const physGraph = createGraph(THREE, 'physics', Y.physF, Y.physB, lane); world.add(physGraph.group);

// loss orbs
const orbG = new THREE.IcosahedronGeometry(1, 2);
const orbData = new THREE.Mesh(orbG, new THREE.MeshBasicMaterial({ color: COLORS.points })), orbPhys = new THREE.Mesh(orbG, new THREE.MeshBasicMaterial({ color: COLORS.physics }));
orbData.position.copy(LOSS.data); orbPhys.position.copy(LOSS.phys); world.add(orbData, orbPhys);

// v lane: a thin tube with gold dashes flowing toward the data graph's backward start
const laneCurve = new THREE.QuadraticBezierCurve3(lane.a, lane.c, lane.b);
const laneG = new THREE.TubeGeometry(laneCurve, 48, 0.045, 6, false);
const laneM = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, uniforms: { uA: { value: 0 }, uT: { value: 0 } },
  vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }',
  fragmentShader: 'varying vec2 vUv; uniform float uA, uT; void main(){ float d = smoothstep(0.35,0.0,abs(fract(vUv.x*7.0 - uT)-0.5)); vec3 c = mix(vec3(0.18,0.82,0.9), vec3(0.98,0.8,0.3), d); gl_FragColor = vec4(c, uA*(0.55+0.45*d)); }' });
const laneMesh = new THREE.Mesh(laneG, laneM); laneMesh.frustumCulled = false; world.add(laneMesh);

// faint conduits (flow direction cues); brightness driven by the phase
const conduits = [];
const addLink = (pts, color) => {
  const g = new THREE.BufferGeometry().setFromPoints(pts), m = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.25 });
  const l = new THREE.Line(g, m); world.add(l); conduits.push({ l, g, m, color: new THREE.Color(color) }); return conduits[conduits.length - 1];
};
const linkOut = addLink([new THREE.Vector3(NET.x1 + 0.4, NET.y, 0), new THREE.Vector3(OUT_X - PACK.w * OUT_S / 2 - 0.1, 0.1, 0)], COLORS.activations);
const linkLossD = addLink([new THREE.Vector3(OUT_X - 1.0, -0.9, 0), LOSS.data.clone()], COLORS.points);
const linkLossP = addLink([new THREE.Vector3(OUT_X - 1.0, -0.9, 0), LOSS.phys.clone()], COLORS.physics);
const linkBwdD = addLink([LOSS.data.clone(), new THREE.Vector3(GR.x1 + 0.2, Y.dataB, 0)], COLORS.gradients);
const linkBwdP = addLink([LOSS.phys.clone(), new THREE.Vector3(GR.x1 + 0.2, Y.physB, 0)], COLORS.physics);

// labels: text is created once (a phone swaps two strings); position = anchor + offset, where the offset is given in SCREEN orientation
const labels = [];
const label = (text, ax, ay, az, dx, dy, opts = {}) => {
  const s = makeLabel(THREE, text, { size: 0.5, ...opts }); s.userData.s0 = s.scale.clone(); s.userData.a = [ax, ay, az]; s.userData.o = [dx, dy]; s.userData.po = opts.po || [dx, dy]; s.userData.short = opts.short;
  world.add(s); labels.push(s); return s;
};
label('battery pack', IN_X, 0, 0, 0, -1.4);
label('heat sink', IN_X, 0, 0, 0, PACK.finTop * IN_S + 0.7);
label('sampled points', IN_X, 0, 0, 7.8, 2.9, { color: '#d8dee9', po: [0, -2.3] });
label('neural network', (NET.x0 + NET.x1) / 2, 0, 0, 0, 2.6);
label('illustrative output', OUT_X, 0, 0, 0, PACK.finTop * OUT_S + 0.7, { color: '#c8d4f0' });
const labelData = label('regular training graph', (GR.x0 + GR.x1) / 2, Y.dataF, 0, 0, 0.7, { color: '#7bd8c4' });
const labelPhys = label('physics graph (derivative of derivative)', (GR.x0 + GR.x1) / 2, Y.physF, 0, 0, 1.9, { color: '#f58ac0' });
label('loss', LOSS.data.x, LOSS.data.y, 0, 1.0, -0.65, { size: 0.42 });
const labelV = label('v = dL/d(u-hat), closed form', 8.9, -6.4, 1.0, 0, 0, { color: '#6fe0f0', size: 0.46 });

// ---------- knobs / state ----------
const knobs = createKnobs($('knob-panel'));
const S = { u: 0, playing: !reduced, s: 0, sTarget: 0, sTween: false, cfg: null, ends: null, rebuildPending: true, hudKey: '' };
const eOf = () => easeInOutCubic(S.s);
if (reduced) S.u = 0.62;
knobs.on('*', () => { S.rebuildPending = true; dirty = true; });

function rebuild() {
  S.rebuildPending = false;
  const cfg = paramsFromKnobs(knobs.get); S.cfg = cfg; S.ends = sageEndpoints(cfg);
  net.rebuild(cfg); dataGraph.rebuild(cfg); physGraph.rebuild(cfg);
  cloud.setCounts(drawCount(cfg.batch), drawCount(cfg.resPoints)); cloud.setLambda(cfg.lambda);
  physGraph.uni.uLambda.value = clamp(0.55 + 0.45 * Math.sqrt(cfg.lambda), 0.4, 1.7);
  S.counts = { data: drawCount(cfg.batch), res: drawCount(cfg.resPoints), nodes: dataGraph.stats.nNodes + physGraph.stats.nNodes };
}

// ---------- layout / camera fit ----------
function reserves() {
  const W = innerWidth, H = innerHeight, top = $('hud-top').getBoundingClientRect().bottom + 4, pr = $('knob-panel').getBoundingClientRect();
  const bot = H - $('hud-bottom').getBoundingClientRect().top + 4 + (mobile() && $('knob-panel').classList.contains('collapsed') ? 52 : 0), right = !mobile() && pr.width > 0 ? Math.max(0, W - pr.left + 6) : 0;
  return { W, H, top, bot, right, left: 0 };
}
function applyViewOffset() {
  const r = reserves();
  camera.setViewOffset(r.W, r.H, (r.right - r.left) / 2, (r.bot - r.top) / 2, r.W, r.H);
  camera.updateProjectionMatrix();
}
function layoutWorld() {
  const pt = innerHeight > innerWidth * 1.15 && mobile();
  if (pt === portrait && world.userData.init) return;
  portrait = pt; world.userData.init = true;
  world.rotation.z = pt ? -Math.PI / 2 : 0;
  const rc = pt ? new THREE.Vector3(CENTER.y, -CENTER.x, 0) : CENTER; // rotated centre
  world.position.copy(CENTER).sub(rc);
  for (const g of [input.group, output.group, cloud.group]) g.rotation.z = pt ? Math.PI / 2 : 0; // packs stay upright on a phone
  for (const [l, a, b] of [[labelPhys, 'physics graph', 'physics graph (derivative of derivative)'], [labelData, 'data graph', 'regular training graph']]) { const t = pt ? a : b; if (l.userData.text !== t) { setLabel(l, t); l.userData.s0.copy(l.scale); } }
  const k = pt ? 1.7 : 1;
  for (const l of labels) {
    const [ax, ay, az] = l.userData.a, [dx, dy] = pt ? l.userData.po : l.userData.o;
    l.position.set(ax + (pt ? -dy : dx), ay + (pt ? dx : dy), az); // screen-frame offset -> world-group frame
    l.scale.copy(l.userData.s0).multiplyScalar(k);
  }
  world.updateMatrixWorld(true);
  const t = new THREE.Vector3(NET.x0 - 0.3, NET.y, 0); world.localToWorld(t); cloud.group.worldToLocal(t); cloud.setTarget(t);
}
function fit() {
  layoutWorld();
  const r = reserves(), uw = Math.max(120, r.W - r.right - r.left), uh = Math.max(120, r.H - r.top - r.bot);
  const bw = portrait ? BOX.max.y - BOX.min.y : BOX.max.x - BOX.min.x, bh = portrait ? BOX.max.x - BOX.min.x : BOX.max.y - BOX.min.y;
  const s = Math.min(uw / bw, uh / (bh + 1.5)) * 0.9;
  const dist = r.H / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * s);
  const az = THREE.MathUtils.degToRad(portrait ? 0 : 14), el = THREE.MathUtils.degToRad(portrait ? 14 : 24);
  camera.position.set(CENTER.x + dist * Math.sin(az) * Math.cos(el), CENTER.y + dist * Math.sin(el), CENTER.z + dist * Math.cos(az) * Math.cos(el));
  controls.target.copy(CENTER); controls.update(); userMoved = false;
}
function resize(refit) {
  const W = innerWidth, H = innerHeight, pr = Math.min(devicePixelRatio || 1, mobile() ? 1.5 : 2);
  renderer.setPixelRatio(pr); renderer.setSize(W, H, false); camera.aspect = W / H;
  cloud.setViewport(H * pr, THREE.MathUtils.degToRad(camera.fov), pr);
  if (refit || !userMoved) fit();
  applyViewOffset(); dirty = true;
}
controls.addEventListener('start', () => { userMoved = true; dirty = true; });
controls.addEventListener('change', () => { dirty = true; });
addEventListener('resize', () => resize(false));

// ---------- UI wiring ----------
const el = { mem: $('r-mem'), time: $('r-time'), nodes: $('r-nodes'), mm: $('r-mm'), phase: $('phase'), play: $('btn-play'), replay: $('btn-replay'), scrub: $('scrub'),
  sage: $('sage-btn'), sageScrub: $('sage-scrub'), panel: $('knob-panel'), ptoggle: $('knob-toggle'), sageState: $('sage-state') };
const setPlaying = (p) => { S.playing = p; el.play.textContent = p ? 'Pause' : 'Play'; el.play.setAttribute('aria-pressed', String(!p)); dirty = true; };
setPlaying(S.playing);
el.play.onclick = () => { if (!S.playing && S.u >= 1) S.u = 0; setPlaying(!S.playing); };
el.replay.onclick = () => { S.u = 0; setPlaying(true); };
el.scrub.oninput = () => { S.u = +el.scrub.value / 1000; setPlaying(false); };
const setSage = (on) => { S.sTarget = on ? 1 : 0; S.sTween = true; if (reduced) { S.s = S.sTarget; S.sTween = false; } dirty = true; };
el.sage.onclick = () => setSage(S.sTarget < 0.5);
el.sageScrub.oninput = () => { S.s = +el.sageScrub.value / 1000; S.sTarget = S.s > 0.5 ? 1 : 0; S.sTween = false; dirty = true; };
addEventListener('keydown', (ev) => {
  if (ev.target.matches?.('input,select,textarea')) return;
  if (ev.code === 'Space') { ev.preventDefault(); el.play.click(); } else if (ev.key === 'r' || ev.key === 'R') el.replay.click(); else if (ev.key === 's' || ev.key === 'S') el.sage.click();
});
const setPanel = (open) => { el.panel.classList.toggle('collapsed', !open); el.ptoggle.setAttribute('aria-expanded', String(open)); requestAnimationFrame(() => { applyViewOffset(); dirty = true; }); };
el.ptoggle.onclick = () => setPanel(el.panel.classList.contains('collapsed'));
setPanel(!mobile());

// ---------- per-frame ----------
function numbers(e) {
  const b = S.ends.base, g = S.ends.sage;
  return { mem: lerp(b.memory, g.memory, e), time: lerp(b.stepTime, g.stepTime, e), nodes: Math.round(lerp(b.nodes, g.nodes, e)), mm: Math.round(lerp(b.matmuls, g.matmuls, e)) };
}
function applyFrame(dt) {
  if (S.rebuildPending) rebuild();
  if (S.sTween) {
    const d = S.sTarget > S.s ? 1 : -1; S.s = clamp(S.s + d * dt / 3.6);
    if (S.s === S.sTarget) S.sTween = false;
    el.sageScrub.value = Math.round(S.s * 1000);
  }
  const e = eOf(), tl = timeline(S.ends, e);
  if (S.playing) { S.u += dt / tl.total; if (S.u >= 1.0) { S.u = 0; } }
  const u = S.u, cfg = S.cfg;

  // points -> network -> output -> loss
  cloud.setFly(prog(u, tl.pts));
  const fp = prog(u, tl.fwd), outP = prog(u, tl.out);
  const bp = prog(u, tl.netBwd), op = prog(u, tl.opt);
  net.setWaves(u >= tl.fwd[0] ? (fp >= 1 ? 1.08 : fp) : 0, u >= tl.netBwd[0] ? (bp >= 1 ? 1.08 : bp) : 0, op > 0 && op < 1 ? Math.sin(Math.PI * op) : 0, cfg.adam);
  output.setReveal(outP); output.packMat.uniforms.uPhase.value = 0;
  // graphs
  for (const g of [dataGraph, physGraph]) {
    g.uni.uU.value = u; g.uni.uFw.value.set(tl.fwd[0], tl.fwd[1]);
  }
  dataGraph.uni.uBw.value.set(tl.bwdData[0], tl.bwdData[1]);
  physGraph.uni.uBw.value.set(tl.bwdPhys[0], tl.bwdPhys[1]); physGraph.uni.uSage.value = e;
  // loss orbs + lane
  const lossGlow = outP >= 1 ? 1 : outP;
  orbData.scale.setScalar(0.22 + 0.1 * lossGlow);
  const pw = Math.sqrt(cfg.lambda);
  orbPhys.scale.setScalar((0.2 + 0.1 * lossGlow) * clamp(pw, 0.5, 1.8) * (1 - 0.35 * e));
  orbPhys.material.color.set(COLORS.physics).lerp(_gold, smoothstepE(e));
  laneM.uniforms.uA.value = clamp((e - 0.25) / 0.4); laneM.uniforms.uT.value = (performance.now() / 1000 * 0.9) % 1;
  labelPhys.visible = e < 0.7; labelPhys.material.opacity = clamp(1 - e * 1.5);
  labelV.visible = e > 0.6; labelV.material.opacity = clamp((e - 0.6) / 0.35);
  // conduits
  linkOut.m.opacity = 0.2 + 0.6 * (fp > 0 && fp < 1 ? 1 : 0);
  for (const l of [linkLossD, linkLossP]) l.m.opacity = 0.2 + 0.5 * (outP > 0 && outP < 1 ? 1 : 0);
  linkBwdD.m.opacity = 0.2 + 0.6 * (u >= tl.bwdData[0] && u < tl.bwdData[1] ? 1 : 0);
  linkBwdP.m.opacity = (0.2 + 0.6 * (u >= tl.bwdPhys[0] && u < tl.bwdPhys[1] ? 1 : 0)) * (1 - smoothstepE(e));
  linkBwdP.l.visible = e < 0.98;
  // HUD
  const n = numbers(e);
  const key = [fmtBytes(n.mem), fmtTime(n.time), n.nodes, n.mm].join('|');
  if (key !== S.hudKey) {
    S.hudKey = key; el.mem.textContent = fmtBytes(n.mem); el.time.textContent = fmtTime(n.time); el.nodes.textContent = n.nodes; el.mm.textContent = n.mm;
  }
  el.mem.dataset.v = n.mem; el.time.dataset.v = n.time; // exact values for tests / tooling
  const ph = phaseAt(u, tl, e); if (el.phase.textContent !== ph) el.phase.textContent = ph;
  if (document.activeElement !== el.scrub) el.scrub.value = Math.round(clamp(u) * 1000);
  const on = S.sTarget > 0.5; el.sage.setAttribute('aria-pressed', String(on)); el.sage.classList.toggle('on', on);
  el.sageState.textContent = on ? 'ON' : 'OFF';
  document.body.dataset.sage = on ? '1' : '0';
}
const _gold = new THREE.Color(COLORS.sageAccent);
const smoothstepE = (x) => x * x * (3 - 2 * x);

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  const animating = S.playing || S.sTween || dirty || S.s > 0 && S.s < 1;
  controls.update();
  if (animating || S.rebuildPending) {
    applyFrame(dt); renderer.render(scene, camera); dirty = false;
  }
  requestAnimationFrame(frame);
}

rebuild(); resize(true);
// test / debug hook (read-only snapshot + a few setters)
window.__sage = {
  state: () => ({ u: S.u, s: S.s, sTarget: S.sTarget, playing: S.playing, nodes: +el.nodes.textContent, matmuls: +el.mm.textContent, memory: +el.mem.dataset.v, stepTime: +el.time.dataset.v,
    memText: el.mem.textContent, timeText: el.time.textContent, phase: el.phase.textContent, counts: S.counts, graphNodes: { data: dataGraph.stats.nNodes, phys: physGraph.stats.nNodes },
    mem: { geometries: renderer.info.memory.geometries, textures: renderer.info.memory.textures }, calls: renderer.info.render.calls, reduced }),
  setU: (u) => { S.u = u; setPlaying(false); dirty = true; },
  setS: (s) => { S.s = s; S.sTarget = s > 0.5 ? 1 : 0; S.sTween = false; el.sageScrub.value = Math.round(s * 1000); dirty = true; },
  knobs, FACTS,
};
requestAnimationFrame(frame);
addEventListener('pagehide', () => { for (const o of [input, output, cloud, net, dataGraph, physGraph]) o.dispose(); disposeObject(scene); renderer.dispose(); });
