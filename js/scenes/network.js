// network.js - module "network": stage 3 (MLP drawn in the shared 3D space) and stage 4 (animated forward pass).
// Everything is ILLUSTRATIVE: synthetic seeded weights of the DRAWN (capped) network; the forward pass is a real
// tiny computation on those weights. Parameter counts come from model.js on the TRUE knob sizes.
// Network API (for memory/physics/sage scenes): ctx.modules.network.api - see llmdocs/CONTRACT_NOTES.md "network API".
import * as NM from './network_math.js';

export const NET_LAYOUT = { x0: -0.15, x1: 7.2, outputPort: [8.3, 0, 0], pitch: 0.5 };
const MAX_LAYERS = 14, MAX_NEURONS = 3 + 12 * NM.CAP + 4, MAX_DOTS = 12 * 3, MAX_EDGES = 3 * NM.CAP + 11 * NM.CAP * NM.CAP + NM.CAP * 4 + 64;
const MAX_PACKETS = 2000, FAN_POINTS = 70, LAUNCH = 3.0;
const CH_COL = [0xe69f00, 0x56b4e9, 0x009e73, 0xcc79a7]; // u v w p (Okabe-Ito)
const CH_NAME = ['u', 'v', 'w', 'p'];

export default function createNetwork(ctx) {
  const { THREE, knobs, util, hud, model, theme } = ctx;
  const group = new THREE.Group(); group.name = 'network';
  let disposed = false;
  const geo = () => ctx.modules?.geometry?.api;

  knobs.define({ id: 'activation', label: 'Activation', type: 'select', options: ['tanh', 'sin', 'silu'], value: 'tanh', group: 'Network', help: 'Activation of the hidden layers (illustrative network).' });

  // ---------------- materials / meshes (fixed capacity: rebuilds never allocate GPU resources) ----------------
  const sphere = new THREE.SphereGeometry(1, 12, 8);
  const nMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5, metalness: 0.1, transparent: true });
  const neurons = new THREE.InstancedMesh(sphere, nMat, MAX_NEURONS + MAX_DOTS); neurons.frustumCulled = false; neurons.count = 0; neurons.renderOrder = 3;
  neurons.setColorAt(0, new THREE.Color(1, 1, 1)); // allocate instanceColor
  const hMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, opacity: 0.35 });
  const halos = new THREE.InstancedMesh(sphere, hMat, MAX_NEURONS); halos.frustumCulled = false; halos.count = 0; halos.renderOrder = 4;
  halos.setColorAt(0, new THREE.Color(0, 0, 0));
  const cyl = new THREE.CylinderGeometry(1, 1, 1, 5, 1, true);
  const eMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55, depthWrite: false, side: THREE.DoubleSide });
  const edges = new THREE.InstancedMesh(cyl, eMat, MAX_EDGES); edges.frustumCulled = false; edges.count = 0; edges.renderOrder = 1;
  edges.setColorAt(0, new THREE.Color(1, 1, 1));
  const pMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95, depthWrite: false });
  const packets = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 6, 4), pMat, MAX_PACKETS); packets.frustumCulled = false; packets.count = 0; packets.renderOrder = 5;
  packets.setColorAt(0, new THREE.Color(1, 1, 1));
  const mkLines = (nSeg, op) => {
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(nSeg * 6), 3)); g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(nSeg * 6), 3));
    const m = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: op, depthWrite: false }); m.userData.baseOpacity = op;
    const l = new THREE.LineSegments(g, m); l.frustumCulled = false; l.renderOrder = 1; g.setDrawRange(0, 0); return l;
  };
  const fan = mkLines(FAN_POINTS * 3, 0.32), outLines = mkLines(4, 0.8);
  group.add(edges, fan, outLines, neurons, halos, packets);
  group.visible = false;

  // labels (pool; each setLabel swaps its own canvas texture, so counts stay constant)
  const labels = [];
  const mkLab = (text, size, color, bg) => { const s = util.makeLabel(THREE, text, { size, color, ...(bg ? { bg } : {}) }); s.visible = false; group.add(s); labels.push(s); return s; };
  const layerLab = Array.from({ length: MAX_LAYERS }, () => mkLab(' ', 0.3, '#dbe8ff'));
  const moreLab = Array.from({ length: 12 }, () => mkLab(' ', 0.25, '#f2c14e'));
  const inLab = ['x', 'y', 'z'].map((t) => mkLab(t, 0.3, '#e8edf5'));
  const outLab = CH_NAME.map((t, i) => mkLab(t, 0.32, '#' + CH_COL[i].toString(16).padStart(6, '0')));
  const headLab = mkLab(' ', 0.38, '#e8edf5'), noteLab = mkLab(' ', 0.3, '#f2c14e'), note2Lab = mkLab(' ', 0.27, '#aeb8cc');
  
  // ---------------- network state ----------------
  const S = {
    sizes: null, net: null, L: 0, off: null, N: 0, nodePos: null, layerX: null, spacing: 1, dots: 0,
    edgeBase: new Float32Array(MAX_EDGES * 16), edgePair: new Uint16Array(MAX_EDGES), E: 0, pairEnd: new Uint32Array(MAX_LAYERS + 1),
    neuronLayer: new Uint8Array(MAX_NEURONS + MAX_DOTS), neuronBase: new Float32Array((MAX_NEURONS + MAX_DOTS) * 3), glow: new Float32Array(MAX_NEURONS + MAX_DOTS), meanAct: new Float32Array(MAX_NEURONS),
    hl: new Float32Array(MAX_LAYERS), hlCol: Array.from({ length: MAX_LAYERS }, () => new THREE.Color(0xf2c14e)),
    P: 0, acts: null, path: new Uint8Array(MAX_PACKETS * MAX_LAYERS), delay: new Float32Array(MAX_PACKETS), idx: new Uint32Array(MAX_PACKETS),
    src: new Float32Array(MAX_PACKETS * 3), srcCol: new Float32Array(MAX_PACKETS * 3), segT: 0.3, lastSeg: new Int16Array(MAX_PACKETS), lastTau: new Float32Array(MAX_PACKETS), version: 0, key: '',
  };
  const listeners = new Set();
  const col = new THREE.Color(), m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), vA = new THREE.Vector3(), vB = new THREE.Vector3(), vS = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0);
  const BASE_IN = new THREE.Color(0xcfd8e8), BASE_HID = new THREE.Color(0x4a90e2), LIT = new THREE.Color(0x6ff0c0), POS = new THREE.Color(0x4a90e2), NEG = new THREE.Color(0xa7b0c4);
  const ps = { clock: 0, playing: true, done: false, arrived: 0, launched: 0, progress: 0, consumed: 0, active: false, fade: 0, reduced: false };

  function nodeIndex(l, k) { return S.off[l] + k; }

  function rebuild() {
    const width = knobs.get('width'), depth = knobs.get('depth'), act = knobs.get('activation');
    const sz = NM.layerSizes(width, depth); S.sizes = sz; const shown = sz.shown, L = shown.length; S.L = L;
    S.net = NM.makeWeights(shown, act); S.off = NM.offsets(shown); S.N = S.off[L];
    const dx = (NET_LAYOUT.x1 - NET_LAYOUT.x0) / (L - 1); S.spacing = dx;
    S.nodePos = new Float32Array(S.N * 3); S.layerX = new Float32Array(L);
    const pitch = NET_LAYOUT.pitch; let nd = 0; const dotsAt = [];
    for (let l = 0; l < L; l++) {
      const n = shown[l], x = NET_LAYOUT.x0 + l * dx; S.layerX[l] = x;
      const { rows, cols } = l === 0 || l === L - 1 ? { rows: n, cols: 1 } : NM.gridDims(n);
      const py = l === 0 ? 0.75 : l === L - 1 ? 0.6 : pitch;
      for (let k = 0; k < n; k++) {
        const row = k % rows, c = Math.floor(k / rows), g = S.off[l] + k;
        S.nodePos[g * 3] = x; S.nodePos[g * 3 + 1] = ((rows - 1) / 2 - row) * py; S.nodePos[g * 3 + 2] = (c - (cols - 1) / 2) * pitch;
        S.neuronLayer[g] = l;
        const bc = l === 0 ? BASE_IN : l === L - 1 ? col.setHex(CH_COL[k]) : BASE_HID; S.neuronBase.set([bc.r, bc.g, bc.b], g * 3);
      }
      if (l > 0 && l < L - 1 && sz.true[l] > n) { const yb = -(rows * py) / 2 - 0.16; for (let j = 0; j < 3; j++) dotsAt.push([x, yb - j * 0.11, 0, l]); }
    }
    S.dots = dotsAt.length; dotsAt.forEach((d, j) => { S.neuronLayer[S.N + j] = d[3]; S.neuronBase.set([BASE_HID.r, BASE_HID.g, BASE_HID.b], (S.N + j) * 3); });
    S._dotPos = dotsAt;
    // edges (thickness and brightness proportional to |w|)
    let e = 0;
    for (let l = 0; l < L - 1; l++) {
      S.pairEnd[l] = e; const nin = shown[l], nout = shown[l + 1], w = S.net.W[l];
      for (let j = 0; j < nout; j++) for (let i = 0; i < nin; i++) {
        const wv = w[j * nin + i], m = NM.weightMag(wv, nin, l === 0 ? 1.6 : S.net.gain);
        const a = (S.off[l] + i) * 3, b = (S.off[l + 1] + j) * 3;
        vA.set(S.nodePos[a], S.nodePos[a + 1], S.nodePos[a + 2]); vB.set(S.nodePos[b], S.nodePos[b + 1], S.nodePos[b + 2]);
        vS.subVectors(vB, vA); const len = vS.length(); vS.multiplyScalar(1 / len); q.setFromUnitVectors(up, vS);
        const r = 0.006 + 0.03 * m; m4.compose(vA.add(vB).multiplyScalar(0.5), q, vB.set(r, len, r)); m4.toArray(S.edgeBase, e * 16);
        const c = (wv >= 0 ? POS : NEG); const k = 0.18 + 0.82 * m; edges.setColorAt(e, col.setRGB(c.r * k, c.g * k, c.b * k));
        S.edgePair[e] = l; e++;
      }
    }
    S.pairEnd[L - 1] = e; S.E = e; edges.count = e; edges.instanceColor.needsUpdate = true; S.edgesDirty = true;
    neurons.count = S.N + S.dots; halos.count = S.N;
    // packets: subset of the input cloud, real forward pass of the drawn net
    const g = geo(), ip = g?.getInputPoints?.(); const count = ip?.count ?? 0;
    const P = Math.max(1, Math.min(MAX_PACKETS, count || 300, util.clamp(Math.round(knobs.get('batch') / 4), 300, MAX_PACKETS))); S.P = P;
    const inputs = new Float32Array(P * 3), rnd = NM.mulberry32(99);
    for (let i = 0; i < P; i++) {
      const id = ip ? Math.floor(i * count / P) : i; S.idx[i] = id;
      for (let a = 0; a < 3; a++) {
        const w = ip ? ip.positions[id * 3 + a] : (rnd() * 2 - 1) * 1.3, cc = ip ? ip.center.getComponent(a) : 0;
        inputs[i * 3 + a] = ip ? (w - cc) / ip.half : w / 1.3; S.src[i * 3 + a] = ip ? w : -2.7 + w;
        S.srcCol[i * 3 + a] = ip ? ip.colors[id * 3 + a] : 0.8;
      }
      S.delay[i] = ip ? ip.rand[id] * LAUNCH : rnd() * LAUNCH;
    }
    const fw = NM.forward(shown, S.net, inputs, P, S.acts); S.acts = fw.acts;
    S.meanAct.fill(0);
    for (let i = 0; i < P; i++) {
      for (let l = 0; l < L; l++) for (let k = 0; k < shown[l]; k++) S.meanAct[S.off[l] + k] += Math.abs(S.acts[i * S.N + S.off[l] + k]) / P;
      for (let l = 0; l < L; l++) S.path[i * MAX_LAYERS + l] = NM.sampleByMagnitude(S.acts, i * S.N + S.off[l], shown[l], rnd(), 0.05);
    }
    S.segT = util.clamp(2.4 / (L + 1), 0.12, 0.4);
    packets.count = P; S.lastSeg.fill(-1); S.lastTau.fill(-1e9); S.glow.fill(0);
    const keep = ps.done || ps.reduced; ps.arrived = 0; ps.launched = 0; if (!keep) { ps.clock = 0; ps.progress = 0; ps.done = false; } else { ps.done = true; ps.progress = 1; }
    if (ps.active && !keep) restartPass(true);
    buildFan(ip); layoutLabels(); S.version++; fanKey = ''; S.key = '';
    for (const fn of listeners) { try { fn(api.getLayerInfo()); } catch (err) { console.error('network listener', err); } }
  }

  // input fan: cloud points -> 3 input neurons, output layer -> output port
  let fanKey = '';
  function buildFan(ip) {
    const pos = fan.geometry.attributes.position.array, cl = fan.geometry.attributes.color.array; let n = 0;
    if (ip) {
      const step = Math.max(1, Math.floor(ip.count / FAN_POINTS)), nin = Math.min(FAN_POINTS, Math.floor(ip.count / step));
      for (let s = 0; s < nin; s++) {
        const id = s * step;
        for (let k = 0; k < 3; k++) {
          const g = k * 3; pos.set([ip.positions[id * 3], ip.positions[id * 3 + 1], ip.positions[id * 3 + 2], S.nodePos[g], S.nodePos[g + 1], S.nodePos[g + 2]], n * 6);
          for (let v = 0; v < 2; v++) { const f = v === 0 ? 1 : 0.35; cl.set([ip.colors[id * 3] * f, ip.colors[id * 3 + 1] * f, ip.colors[id * 3 + 2] * f], n * 6 + v * 3); }
          n++;
        }
      }
    }
    fan.geometry.setDrawRange(0, n * 2); fan.geometry.attributes.position.needsUpdate = true; fan.geometry.attributes.color.needsUpdate = true;
    const op = outLines.geometry.attributes.position.array, oc = outLines.geometry.attributes.color.array, P0 = NET_LAYOUT.outputPort;
    for (let k = 0; k < 4; k++) {
      const g = (S.off[S.L - 1] + k) * 3; op.set([S.nodePos[g], S.nodePos[g + 1], S.nodePos[g + 2], P0[0], P0[1] * 0 + S.nodePos[g + 1] * 0.25, P0[2]], k * 6);
      const c = col.setHex(CH_COL[k]); oc.set([c.r, c.g, c.b, c.r * 0.3, c.g * 0.3, c.b * 0.3], k * 6);
    }
    outLines.geometry.setDrawRange(0, 8); outLines.geometry.attributes.position.needsUpdate = true; outLines.geometry.attributes.color.needsUpdate = true;
  }

  // labels
  let labelVis = new Float32Array(MAX_LAYERS);
  function layoutLabels() {
    const { sizes: sz, L, spacing } = S, act = knobs.get('activation'), truncated = sz.true[1] > sz.shown[1];
    const fit = (chars, stagger) => (chars * 0.175 + 0.25) <= (stagger ? 2 : 1) * spacing * 0.95;
    const variants = (l) => (l === 0 ? ['Input (x, y, z)', 'Input', 'In'] : l === L - 1 ? [`Linear ${sz.true[l - 1]}→${sz.true[l]}`, `Linear ${sz.true[l]}`, 'Out'] : [`Linear ${sz.true[l - 1]}→${sz.true[l]} + ${act}`, `${sz.true[l]} + ${act}`, `${sz.true[l]}`]);
    let pick = null;
    const tries = [[0, 1], [1, 1], [2, 1], [2, 2], [2, 3], [2, 4]];
    for (const [vi, stride] of tries) {
      const maxChars = Math.max(...Array.from({ length: L }, (_, l) => variants(l)[vi].length)); // conservative
      if (fit(maxChars, stride === 1) || stride === 4) { pick = [vi, stride]; break; }
    }
    const [vi, stride] = pick;
    layerLab.forEach((s, l) => {
      if (l >= L) { s.userData.want = false; return; }
      const show = l % stride === 0 || l === L - 1; s.userData.want = show;
      if (show) { util.setLabel(s, variants(l)[vi]); s.position.set(S.layerX[l], 1.5 + (stride === 1 && l % 2 ? 0.34 : 0), 0); s.userData.layer = l; }
    });
    moreLab.forEach((s, i) => {
      const l = i + 1, ok = truncated && l < L - 1 && spacing >= 1.15; s.userData.want = ok; s.userData.layer = l;
      if (ok) { util.setLabel(s, `… ×${sz.true[l] - sz.shown[l]} more`); const rows = NM.gridDims(sz.shown[l]).rows; s.position.set(S.layerX[l], -(rows * NET_LAYOUT.pitch) / 2 - 0.55, 0); }
    });
    inLab.forEach((s, k) => { s.userData.want = true; s.userData.layer = 0; const g = k * 3; s.position.set(S.nodePos[g] - 0.32, S.nodePos[g + 1], S.nodePos[g + 2]); });
    outLab.forEach((s, k) => { s.userData.want = true; s.userData.layer = L - 1; const g = (S.off[L - 1] + k) * 3; s.position.set(S.nodePos[g] + 0.3, S.nodePos[g + 1], S.nodePos[g + 2]); });
    const cx = (NET_LAYOUT.x0 + NET_LAYOUT.x1) / 2;
    util.setLabel(headLab, `Illustrative MLP  3 → ${sz.true[1]} ×${knobs.get('depth')} → 4  ·  ${act}`); headLab.position.set(cx, 3.0, 0);
    util.setLabel(noteLab, truncated ? `drawing ${sz.shown[1]} of ${sz.true[1]} neurons per hidden layer (… ×${sz.true[1] - sz.shown[1]} more each, not drawn)` : `all ${sz.true[1]} neurons per hidden layer drawn`);
    noteLab.position.set(cx, -2.45, 0);
    util.setLabel(note2Lab, 'synthetic seeded weights (illustrative) · edge thickness ∝ |w| · blue +, grey −'); note2Lab.position.set(cx, -2.85, 0);
    for (const s of [headLab, noteLab, note2Lab]) { s.userData.want = true; s.userData.layer = -1; }
  }

  // ---------------- forward pass control ----------------
  function restartPass(silent) {
    ps.clock = 0; ps.done = false; ps.arrived = 0; ps.launched = 0; ps.progress = 0; S.lastSeg.fill(-1); S.lastTau.fill(-1e9); S.glow.fill(0);
    if (ps.reduced) { ps.clock = 1e3; ps.done = true; ps.progress = 1; }
    geo()?.setOutputProgress?.(ps.progress); geo()?.setInputConsumed?.(ps.reduced ? 0 : 0);
    ps.consumed = 0; if (!silent) ps.playing = true; syncUI();
  }

  // overlay controls (DOM)
  const ui = document.createElement('div'); ui.id = 'net-controls'; ui.className = 'card'; ui.setAttribute('role', 'group'); ui.setAttribute('aria-label', 'Forward pass controls');
  Object.assign(ui.style, { display: 'none', top: '86px', left: '12px', padding: '6px 8px', gap: '6px', alignItems: 'center', zIndex: 5 });
  const mkBtn = (txt, label) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = txt; b.setAttribute('aria-label', label); Object.assign(b.style, { background: '#121a2b', color: '#e8edf5', border: '1px solid #26314a', borderRadius: '8px', padding: '4px 10px', font: 'inherit', cursor: 'pointer' }); return b; };
  const bReplay = mkBtn('↻ Replay', 'Replay forward pass'), bPlay = mkBtn('❚❚ Pause', 'Pause or play forward pass');
  const uiNote = document.createElement('span'); Object.assign(uiNote.style, { color: '#8793a6', fontSize: '12px' });
  ui.append(bReplay, bPlay, uiNote); document.body.appendChild(ui);
  bReplay.onclick = () => api.replay(); bPlay.onclick = () => api.setPlaying(!ps.playing);
  function syncUI() { bPlay.textContent = ps.playing ? '❚❚ Pause' : '▶ Play'; uiNote.textContent = `${S.P?.toLocaleString('en') ?? 0} packets · illustrative`; }
  const showUI = (on) => { ui.style.display = on ? 'flex' : 'none'; };

  // ---------------- public API ----------------
  const api = {
    NET_LAYOUT,
    /** Array of {index, kind:'input'|'hidden'|'output', x, center:[x,y,z], trueSize, shown, rows, cols, size:[sy,sz], label, activation, paramsIn}. */
    getLayerInfo() {
      return Array.from({ length: S.L }, (_, l) => {
        const n = S.sizes.shown[l], d = l === 0 || l === S.L - 1 ? { rows: n, cols: 1 } : NM.gridDims(n);
        return { index: l, kind: l === 0 ? 'input' : l === S.L - 1 ? 'output' : 'hidden', x: S.layerX[l], center: [S.layerX[l], 0, 0], trueSize: S.sizes.true[l], shown: n, rows: d.rows, cols: d.cols,
          size: [d.rows * NET_LAYOUT.pitch, d.cols * NET_LAYOUT.pitch], label: l === 0 ? 'Input (x, y, z)' : `Linear ${S.sizes.true[l - 1]}→${S.sizes.true[l]}` + (l < S.L - 1 ? ' + ' + knobs.get('activation') : ''),
          activation: l === 0 || l === S.L - 1 ? null : knobs.get('activation'), paramsIn: l === 0 ? 0 : S.sizes.true[l - 1] * S.sizes.true[l] + S.sizes.true[l] };
      });
    },
    get layerCount() { return S.L; },
    /** World anchor (THREE.Vector3, fresh) of layer l's plane centre; optional dy/dz offsets. */
    getLayerAnchor(l, dy = 0, dz = 0) { l = Math.max(0, Math.min(S.L - 1, l | 0)); return new THREE.Vector3(S.layerX[l], dy, dz); },
    /** World position of drawn neuron k in layer l. */
    getNeuronPosition(l, k) { const g = nodeIndex(l, Math.min(k, S.sizes.shown[l] - 1)) * 3; return new THREE.Vector3(S.nodePos[g], S.nodePos[g + 1], S.nodePos[g + 2]); },
    /** Highlight layer(s): layer = index | 'all' | array; amount 0..1 (0 clears); color hex optional (default gold). */
    setActivationHighlight(layer, amount = 1, color = 0xf2c14e) {
      const set = (l) => { if (l >= 0 && l < MAX_LAYERS) { S.hl[l] = util.clamp(amount); S.hlCol[l].setHex(color); } };
      if (layer === 'all') for (let l = 0; l < MAX_LAYERS; l++) set(l); else if (Array.isArray(layer)) layer.forEach(set); else set(layer | 0);
    },
    clearHighlights() { S.hl.fill(0); },
    /** Subscribe to rebuilds (width/depth/activation/batch changed). fn(layerInfo[]). Returns unsubscribe. */
    onRebuild(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    get version() { return S.version; },
    get params() { return model.paramCount(ctx.params()); },
    get drawn() { return { perHidden: S.sizes.shown[1], trueWidth: S.sizes.true[1], depth: S.L - 2, packets: S.P }; },
    replay() { ps.playing = true; restartPass(); }, setPlaying(p) { ps.playing = !!p; syncUI(); },
    get state() { return { playing: ps.playing, done: ps.done, progress: ps.progress, clock: ps.clock }; },
  };

  // ---------------- camera framing (accounts for HUD, knob panel and aspect) ----------------
  function frame(x0, x1, o = {}) {
    const W = innerWidth, H = innerHeight, mobile = W <= 640, { az = mobile ? -52 : -24, el = mobile ? 24 : 15, yMin = -3.0, yMax = 3.5, zHalf = 1.2 } = o;
    const rect = (id) => document.getElementById(id)?.getBoundingClientRect();
    const top = (rect('hud-top')?.bottom ?? 80) + 8, cardTop = rect('hud-bottom')?.top ?? H - 170;
    let left = 8, right = W - 8, bottom = cardTop - 8;
    if (mobile) bottom = Math.min(bottom, H - 250); else right = W - 330;
    const a = az * Math.PI / 180, e = el * Math.PI / 180, tv = Math.tan(ctx.camera.fov * Math.PI / 360);
    const dir = new THREE.Vector3(Math.sin(a) * Math.cos(e), Math.sin(e), Math.cos(a) * Math.cos(e)), rightV = new THREE.Vector3(Math.cos(a), 0, -Math.sin(a)), upV = new THREE.Vector3().crossVectors(dir, rightV).normalize();
    const c = new THREE.Vector3((x0 + x1) / 2, (yMin + yMax) / 2, 0), cam = new THREE.PerspectiveCamera(ctx.camera.fov, W / H, 0.1, 400), pv = new THREE.Vector3();
    const ux = (left + right) / 2, uy = (top + bottom) / 2, uw = right - left, uh = bottom - top; let d = 30;
    // numerically fit the content box (projected corners) into the usable screen rectangle
    for (let it = 0; it < 6; it++) {
      cam.position.copy(c).addScaledVector(dir, d); cam.lookAt(c); cam.updateMatrixWorld(); cam.updateProjectionMatrix();
      let x0p = 1e9, x1p = -1e9, y0p = 1e9, y1p = -1e9;
      for (const X of [x0, x1]) for (const Y of [yMin, yMax]) for (const Z of [-zHalf, zHalf]) { pv.set(X, Y, Z).project(cam); const sx = (pv.x * 0.5 + 0.5) * W, sy = (0.5 - pv.y * 0.5) * H; x0p = Math.min(x0p, sx); x1p = Math.max(x1p, sx); y0p = Math.min(y0p, sy); y1p = Math.max(y1p, sy); }
      const wpp = 2 * d * tv / H, k = Math.max((x1p - x0p) / uw, (y1p - y0p) / uh) * 1.02;
      c.addScaledVector(rightV, -(ux - (x0p + x1p) / 2) * wpp).addScaledVector(upV, ((uy - (y0p + y1p) / 2)) * wpp);
      d = Math.min(58, d * k);
    }
    const p = c.clone().addScaledVector(dir, d); return { pos: [p.x, p.y, p.z], target: [c.x, c.y, c.z] };
  }

  // ---------------- per-frame ----------------
  let lastRe = -1, lastStamp = -1, readKey = '';
  const rgb = new Float32Array(3);
  function tick(dt, state) {
    if (disposed || lastStamp === state.time) return; lastStamp = state.time;
    const rl = util.revealOf(state, 'network.layers'), re = util.revealOf(state, 'network.edges'), rp = util.revealOf(state, 'forward.pulse'), rf = util.revealOf(state, 'forward.field');
    ps.reduced = state.reducedMotion;
    group.visible = rl > 0.001 || re > 0.001;
    if (!group.visible) { return; }
    const L = S.L, inFwd = state.stageId === 'forward';
    const g = geo();
    // ---- edges reveal (layer by layer)
    if (re !== lastRe || S.edgesDirty) {
      lastRe = re; S.edgesDirty = false; const arr = edges.instanceMatrix.array, base = S.edgeBase;
      const K = new Float32Array(L); for (let l = 0; l < L - 1; l++) K[l] = util.smoothstep((l + 0.3) / L * 0.7, (l + 0.3) / L * 0.7 + 0.3, re);
      for (let e = 0; e < S.E; e++) { const k = K[S.edgePair[e]], o = e * 16; for (let i = 0; i < 16; i++) arr[o + i] = base[o + i]; arr[o] *= k; arr[o + 1] *= k; arr[o + 2] *= k; arr[o + 8] *= k; arr[o + 9] *= k; arr[o + 10] *= k; }
      edges.instanceMatrix.needsUpdate = true; edges.visible = re > 0.001; eMat.opacity = 0.55;
    }
    fan.material.opacity = fan.material.userData.baseOpacity * re; fan.visible = re > 0.01;
    outLines.material.opacity = outLines.material.userData.baseOpacity * util.smoothstep(0.2, 1, rf) * rl; outLines.visible = rf > 0.01 && rl > 0.5;

    // ---- packets / pass state machine
    const wantActive = rp > 0.001 && inFwd; ps.active = inFwd;
    const travel = (L + 1) * S.segT, period = LAUNCH + travel + 0.8;
    ps.fade += ((inFwd ? 1 : 0) - ps.fade) * (1 - Math.exp(-dt * 6)); if (!inFwd && ps.fade < 0.01) ps.fade = 0;
    if (ps.playing && !ps.reduced && (inFwd || ps.fade > 0.01)) ps.clock += dt;
    if (!ps.done && ps.clock >= LAUNCH + travel + 0.05) { ps.done = true; ps.progress = 1; }
    const loop = ps.done;
    if (!inFwd && rf > 0.5 && ps.progress < 1) { ps.done = true; ps.progress = 1; g?.setOutputProgress?.(1); }
    const sz = packets.instanceMatrix.array; let arrived = 0, launched = 0, colDirty = false;
    const P = S.P, N = S.N, Lq = L, pc = packets.instanceColor.array, pr = 0.05;
    const showPk = rp * ps.fade > 0.01 && !ps.reduced;
    packets.visible = showPk;
    for (let i = 0; i < P; i++) {
      const o = i * 16; let tau = (ps.clock - S.delay[i]) / S.segT;
      if (loop) { const ph = ((ps.clock - S.delay[i]) % period + period) % period; tau = ph / S.segT; if (tau < S.lastTau[i]) S.lastSeg[i] = -1; S.lastTau[i] = tau; if (ps.clock - S.delay[i] < 0) tau = -1; }
      else if (tau >= 0) launched++;
      const seg = tau < 0 ? -1 : Math.min(Math.floor(tau), L + 1);
      if (seg >= L + 1 && !loop) arrived++;
      let s = 0, x = 0, y = 0, z = 0;
      if (seg >= 0) {
        // glow on arrival at layer seg-1
        for (let k = S.lastSeg[i] + 1; k <= seg; k++) if (k >= 1 && k <= L && !ps.reduced) { const l = k - 1, nn = S.off[l] + S.path[i * MAX_LAYERS + l]; S.glow[nn] += (70 / P) * (0.3 + Math.min(1, Math.abs(S.acts[i * N + nn]))); }
        if (seg !== S.lastSeg[i]) { S.lastSeg[i] = seg; colDirty = true; { const a = seg >= 1 && seg <= L ? Math.min(1, Math.abs(S.acts[i * N + S.off[seg - 1] + S.path[i * MAX_LAYERS + seg - 1]])) : 0; if (seg === 0) { pc[i * 3] = S.srcCol[i * 3]; pc[i * 3 + 1] = S.srcCol[i * 3 + 1]; pc[i * 3 + 2] = S.srcCol[i * 3 + 2]; } else if (seg <= L) { pc[i * 3] = 0.3 + 0.2 * a; pc[i * 3 + 1] = 0.9 + 0.1 * a; pc[i * 3 + 2] = 0.65 + 0.2 * a; } else { pc[i * 3] = 1; pc[i * 3 + 1] = 0.85; pc[i * 3 + 2] = 0.4; } } }
        const f0 = Math.min(seg, L), u0 = tau - seg, u = u0 * u0 * (3 - 2 * u0);
        // node positions: 0 = cloud point, 1..L = layers, L+1 = port
        let ax, ay, az, bx, by, bz;
        if (seg === 0) { ax = S.src[i * 3]; ay = S.src[i * 3 + 1]; az = S.src[i * 3 + 2]; } else if (seg <= L) { const g0 = (S.off[seg - 1] + S.path[i * MAX_LAYERS + seg - 1]) * 3; ax = S.nodePos[g0]; ay = S.nodePos[g0 + 1]; az = S.nodePos[g0 + 2]; } else { ax = NET_LAYOUT.outputPort[0]; ay = 0; az = 0; }
        if (seg + 1 <= L) { const g1 = (S.off[seg] + S.path[i * MAX_LAYERS + seg]) * 3; bx = S.nodePos[g1]; by = S.nodePos[g1 + 1]; bz = S.nodePos[g1 + 2]; } else { bx = NET_LAYOUT.outputPort[0]; by = ay * 0.25; bz = 0; }
        if (seg >= L + 1) { x = ax; y = ay; z = az; s = Math.max(0, 1 - (tau - (L + 1)) * 3) * pr; }
        else { x = ax + (bx - ax) * u; y = ay + (by - ay) * u; z = az + (bz - az) * u; s = pr * (seg === 0 ? 0.5 + 0.5 * u : 1) * (seg === L ? 1 - 0.5 * u : 1); }
        void f0;
      }
      sz[o] = s; sz[o + 5] = s; sz[o + 10] = s; sz[o + 12] = x; sz[o + 13] = y; sz[o + 14] = z; sz[o + 15] = 1;
      if (seg < 0) { S.lastSeg[i] = loop ? S.lastSeg[i] : -1; }
    }
    if (showPk || packets.visible) { packets.instanceMatrix.needsUpdate = true; if (colDirty) packets.instanceColor.needsUpdate = true; }
    ps.arrived = arrived; ps.launched = launched;
    if (!ps.done && !ps.reduced) ps.progress = arrived / Math.max(1, P);
    if (inFwd || ps.fade > 0.01) {
      if (g) {
        g.setOutputProgress?.(ps.progress);
        const target = loop ? 0 : launched / Math.max(1, P); ps.consumed += (target - ps.consumed) * (loop ? 1 - Math.exp(-dt * 3) : 1);
        g.setInputConsumed?.(ps.reduced ? 0 : ps.consumed);
      }
    }

    // ---- neurons: layer-by-layer reveal + activation glow + highlights
    const na = neurons.instanceMatrix.array, nc = neurons.instanceColor.array, ha = halos.instanceMatrix.array, hc = halos.instanceColor.array, decay = Math.exp(-dt * 4);
    for (let n = 0; n < S.N + S.dots; n++) {
      const l = S.neuronLayer[n], k = util.smoothstep(l / L * 0.7, l / L * 0.7 + 0.3, rl);
      const isDot = n >= S.N; let x, y, z;
      if (isDot) { const d = S._dotPos[n - S.N]; x = d[0]; y = d[1]; z = d[2]; } else { x = S.nodePos[n * 3]; y = S.nodePos[n * 3 + 1]; z = S.nodePos[n * 3 + 2]; }
      let gl = 0;
      if (!isDot) { if (ps.reduced) gl = Math.min(1, S.meanAct[n]) * (inFwd ? 0.8 : 0); else { S.glow[n] *= decay; if (!inFwd && ps.fade < 0.01) S.glow[n] *= 0.9; gl = Math.min(1, S.glow[n]); } }
      const r = (isDot ? 0.035 : 0.11 + (l === 0 || l === L - 1 ? 0.025 : 0)) * k * (1 + 0.45 * gl);
      const o = n * 16; na[o] = r; na[o + 5] = r; na[o + 10] = r; na[o + 12] = x; na[o + 13] = y; na[o + 14] = z; na[o + 15] = 1;
      const b0 = n * 3, hlA = S.hl[l]; const h = S.hlCol[l];
      let cr = S.neuronBase[b0] * (1 - gl) + LIT.r * gl, cg = S.neuronBase[b0 + 1] * (1 - gl) + LIT.g * gl, cb = S.neuronBase[b0 + 2] * (1 - gl) + LIT.b * gl;
      if (hlA > 0) { cr += (h.r - cr) * hlA; cg += (h.g - cg) * hlA; cb += (h.b - cb) * hlA; }
      nc[b0] = cr; nc[b0 + 1] = cg; nc[b0 + 2] = cb;
      if (!isDot) { const ho = n * 16, hr = r * (1.7 + 1.2 * gl), hv = gl * 0.9 + hlA * 0.5; ha[ho] = hr; ha[ho + 5] = hr; ha[ho + 10] = hr; ha[ho + 12] = x; ha[ho + 13] = y; ha[ho + 14] = z; ha[ho + 15] = 1; hc[b0] = LIT.r * hv * (1 - hlA) + h.r * hlA * hv; hc[b0 + 1] = LIT.g * hv * (1 - hlA) + h.g * hlA * hv; hc[b0 + 2] = LIT.b * hv * (1 - hlA) + h.b * hlA * hv; }
    }
    neurons.instanceMatrix.needsUpdate = true; neurons.instanceColor.needsUpdate = true; halos.instanceMatrix.needsUpdate = true; halos.instanceColor.needsUpdate = true;
    neurons.visible = halos.visible = rl > 0.001;

    // ---- labels
    for (const s of labels) {
      if (!s.userData.want) { s.visible = false; continue; }
      const l = s.userData.layer, k = l < 0 ? rl : util.smoothstep(l / L * 0.7, l / L * 0.7 + 0.3, rl);
      s.visible = k > 0.02; s.material.opacity = k;
    }
    // ---- HUD
    if (state.stageId === 'network' || inFwd) {
      const key = state.stageId + S.version + (inFwd ? Math.round(ps.progress * 20) : 0);
      if (key !== readKey) {
        readKey = key; const pc2 = model.paramCount(ctx.params());
        hud.readout('params', 'Parameters (true size)', pc2.toLocaleString('en'), 'model estimate');
        hud.readout('drawn', 'Neurons drawn / layer', `${S.sizes.shown[1]} of ${S.sizes.true[1]}`);
        if (inFwd) hud.readout('pass', 'Forward pass', ps.done ? 'complete' : `${Math.round(ps.progress * 100)}% of ${S.P.toLocaleString('en')} packets`);
      }
    }
    void wantActive; void rgb; void colDirty;
  }

  // ---------------- knobs: live rebuild ----------------
  const offs = ['width', 'depth', 'activation', 'batch'].map((id) => knobs.on(id, () => { if (!disposed) { rebuild(); syncUI(); } }));
  rebuild(); syncUI();

  // mobile (portrait): crop to the network (stage 3) / network + part of cloud and output (stage 4) so the pipeline is not tiny
  const mob = () => innerWidth <= 640, fr3 = () => (mob() ? frame(-1.1, 8.8, { az: -40 }) : frame(-4.4, 12.8)), fr4 = () => (mob() ? frame(-2.8, 13.0, { az: -40 }) : frame(-4.4, 14.8));
  const camNet = { get pos() { return fr3().pos; }, get target() { return fr3().target; } };
  const camFwd = { get pos() { return fr4().pos; }, get target() { return fr4().target; } };
  const reset = () => { readKey = ''; };
  return {
    group, api,
    stages: {
      network: { enter() { reset(); showUI(false); }, exit() {}, update: tick, camera: camNet },
      forward: { enter(t) { reset(); showUI(true); if (ps.reduced) { restartPass(); } else { ps.playing = true; restartPass(); } },
        exit() { showUI(false); geo()?.setInputConsumed?.(0); }, update: tick, camera: camFwd },
    },
    dispose() {
      disposed = true; offs.forEach((f) => f()); listeners.clear(); ui.remove();
      util.disposeObject(group); sphere.dispose(); cyl.dispose(); packets.geometry.dispose();
      [neurons, halos, edges, packets].forEach((m) => m.dispose());
    },
  };
}
