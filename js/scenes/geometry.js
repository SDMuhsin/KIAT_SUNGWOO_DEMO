// geometry.js - module "geometry": stage 1 (duct + empty output slot), stage 2 (sampling -> input cloud), and the
// OUTPUT FIELD display used by stage 4+ (driven through the exported api, see llmdocs/CONTRACT_NOTES.md "geometry API").
// Everything is procedural and ILLUSTRATIVE (synthetic analytic field, not a CFD result).
import * as M from './geometry_math.js';

// World layout (also documented in CONTRACT_NOTES.md)
export const LAYOUT = {
  geomCenter: [-8.5, 0, 0],      // input duct (6 x 2 x 3)
  cloudCenter: [-2.7, 0, 0],     // normalised input cube, half extent 1.3
  cloudHalf: 1.3,
  inputPort: [-1.3, 0, 0],       // right face centre of the cloud: network input layer attaches here
  outCenter: [11.5, 0, 0],       // output copy of the duct
  outputPort: [8.3, 0, 0],       // just left of the output duct inlet face: network output layer ends here
};

export default function createGeometry(ctx) {
  const { THREE, knobs, util, hud } = ctx;
  const group = new THREE.Group(); group.name = 'geometry';
  const { L, H, W } = M.DUCT;
  const state0 = { disposed: false };
  const tmpC = [0, 0, 0];
  const hex = (h) => new THREE.Color(h);

  // ---------- shared builders ----------
  const trackedTex = [];
  function buildDuct(opts) {
    const g = new THREE.Group();
    const wallMat = new THREE.MeshStandardMaterial({ color: 0x8a94a6, transparent: true, opacity: opts.shell, side: THREE.DoubleSide, depthWrite: false, roughness: 0.6, metalness: 0.1, emissive: 0x2a3140, emissiveIntensity: 0.6 });
    wallMat.userData.baseOpacity = opts.shell;
    const mk = (w, h, pos, rot, mat) => { const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat); m.position.set(...pos); m.rotation.set(...rot); g.add(m); return m; };
    mk(L, W, [0, H / 2, 0], [-Math.PI / 2, 0, 0], wallMat); mk(L, W, [0, -H / 2, 0], [-Math.PI / 2, 0, 0], wallMat);
    mk(L, H, [0, 0, W / 2], [0, 0, 0], wallMat); mk(L, H, [0, 0, -W / 2], [0, 0, 0], wallMat);
    const mkEnd = (x, col) => { const m = new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: opts.port, side: THREE.DoubleSide, depthWrite: false }); m.userData.baseOpacity = opts.port; return mk(W, H, [x, 0, 0], [0, Math.PI / 2, 0], m); };
    mkEnd(-L / 2, M.KIND_HEX[2]); mkEnd(L / 2, M.KIND_HEX[3]);
    const box = new THREE.EdgesGeometry(new THREE.BoxGeometry(L, H, W));
    const em = new THREE.LineBasicMaterial({ color: opts.edge, transparent: true, opacity: opts.edgeOp }); em.userData.baseOpacity = opts.edgeOp;
    g.add(new THREE.LineSegments(box, em));
    for (const [x, col] of [[-L / 2, M.KIND_HEX[2]], [L / 2, M.KIND_HEX[3]]]) { // bright port rims
      const p = [[x, -H / 2, -W / 2], [x, H / 2, -W / 2], [x, H / 2, W / 2], [x, -H / 2, W / 2]].map((a) => new THREE.Vector3(...a));
      const lm = new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: opts.edgeOp }); lm.userData.baseOpacity = opts.edgeOp;
      g.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(p), lm));
    }
    // battery-cell-like cylinders (axis along y)
    const cellMat = new THREE.MeshStandardMaterial({ color: opts.cell, transparent: true, opacity: opts.cellOp, roughness: 0.35, metalness: 0.6, emissive: 0x1a2030, emissiveIntensity: 0.5 }); cellMat.userData.baseOpacity = opts.cellOp;
    const bandMat = new THREE.MeshStandardMaterial({ color: opts.band, transparent: true, opacity: opts.cellOp, roughness: 0.5, metalness: 0.2, emissive: opts.band, emissiveIntensity: 0.25 }); bandMat.userData.baseOpacity = opts.cellOp;
    const cg = new THREE.CylinderGeometry(M.R, M.R, H, 40, 1), bg = new THREE.CylinderGeometry(M.R + 0.012, M.R + 0.012, 0.32, 40, 1, true), tg = new THREE.CylinderGeometry(M.R * 0.35, M.R * 0.35, 0.1, 20);
    for (const c of M.CELLS) {
      const cm = new THREE.Mesh(cg, cellMat); cm.position.set(c.x, 0, c.z); g.add(cm);
      const b = new THREE.Mesh(bg, bandMat); b.position.set(c.x, H * 0.26, c.z); g.add(b);
      const t = new THREE.Mesh(tg, bandMat); t.position.set(c.x, H / 2 + 0.04, c.z); g.add(t);
    }
    return g;
  }
  function arrows(col, x0) {
    const g = new THREE.Group(), ma = new THREE.MeshStandardMaterial({ color: col, emissive: col, emissiveIntensity: 0.5, transparent: true, opacity: 0.95, roughness: 0.4 }); ma.userData.baseOpacity = 0.95;
    for (const z of [-0.9, 0, 0.9]) {
      const sh = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.55, 10), ma); sh.rotation.z = -Math.PI / 2; sh.position.set(0.28, 0, z);
      const hd = new THREE.Mesh(new THREE.ConeGeometry(0.15, 0.32, 14), ma); hd.rotation.z = -Math.PI / 2; hd.position.set(0.71, 0, z);
      g.add(sh, hd);
    }
    g.position.set(x0, 0, 0); g.userData.x0 = x0; return g;
  }
  const label = (txt, size, opts) => util.makeLabel(THREE, txt, { size, ...opts });

  // ---------- INPUT duct (stage 1) ----------
  const duct = new THREE.Group(); duct.name = 'duct'; duct.position.set(...LAYOUT.geomCenter);
  duct.add(buildDuct({ shell: 0.1, port: 0.12, edge: 0x9fb0c8, edgeOp: 0.85, cell: 0x5a6a85, cellOp: 0.92, band: 0x4a90e2 }));
  const inA = arrows(M.KIND_HEX[2], -L / 2 - 1.25), outA = arrows(M.KIND_HEX[3], L / 2 + 0.35);
  duct.add(inA, outA);
  const tIn = label('INLET', 0.34, { color: '#00b894' }), tOut = label('OUTLET', 0.34, { color: '#cc79a7' });
  tIn.position.set(-L / 2 - 0.9, 0.95, 0); tOut.position.set(L / 2 + 0.55, 0.8, 0);
  const tCell = label('battery cells', 0.3, { color: '#b9c4d8' }); tCell.position.set(L / 2 - 0.2, -H / 2 - 0.35, W / 2 + 0.2);
  const tTitle = label('Battery-pack cooling duct (illustrative)', 0.38); tTitle.position.set(0, H / 2 + 0.75, 0);
  duct.add(tIn, tOut, tCell, tTitle); group.add(duct);

  // ---------- OUTPUT slot: ghost copy (fieldslot) + field (forward.field) ----------
  const slot = new THREE.Group(); slot.name = 'slot'; slot.position.set(...LAYOUT.outCenter);
  slot.add(buildDuct({ shell: 0.07, port: 0.06, edge: 0x7f8aa0, edgeOp: 0.6, cell: 0x4a566d, cellOp: 0.9, band: 0x3a4a66 }));
  const tSlot = label('Output field slot (empty)', 0.4, { color: '#aeb8cc' }); tSlot.position.set(0, H / 2 + 0.75, 0);
  slot.add(tSlot); group.add(slot);

  // field layer (separately faded, not under gateGroup)
  const field = new THREE.Group(); field.position.set(...LAYOUT.outCenter); field.visible = false; group.add(field);
  const TW = 192, TH = 96, texData = new Uint8Array(TW * TH * 4), truth = new Float32Array(TW * TH * 2), rndT = new Float32Array(TW * TH), mask = new Uint8Array(TW * TH);
  const rr = M.mulberry32(5);
  for (let j = 0; j < TH; j++) for (let i = 0; i < TW; i++) {
    const k = j * TW + i, x = -L / 2 + (i + 0.5) / TW * L, z = W / 2 - (j + 0.5) / TH * W;
    mask[k] = M.insideCell(x, z) ? 1 : 0; truth[k * 2] = M.fieldValue01(x, 0, z, 'speed'); truth[k * 2 + 1] = M.fieldValue01(x, 0, z, 'pressure'); rndT[k] = rr();
  }
  const spots = M.hotspots(), hot = new Float32Array(TW * TH);
  for (let j = 0; j < TH; j++) for (let i = 0; i < TW; i++) hot[j * TW + i] = M.hotspotIntensity(spots, -L / 2 + (i + 0.5) / TW * L, W / 2 - (j + 0.5) / TH * W);
  const sliceTex = new THREE.DataTexture(texData, TW, TH, THREE.RGBAFormat); sliceTex.colorSpace = THREE.SRGBColorSpace; sliceTex.magFilter = THREE.LinearFilter; sliceTex.minFilter = THREE.LinearFilter; sliceTex.needsUpdate = true;
  const sliceMat = new THREE.MeshBasicMaterial({ map: sliceTex, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false });
  const slice = new THREE.Mesh(new THREE.PlaneGeometry(L, W), sliceMat); slice.rotation.x = -Math.PI / 2; field.add(slice);
  // violation overlay on the slice
  const vData = new Uint8Array(TW * TH * 4), vTex = new THREE.DataTexture(vData, TW, TH, THREE.RGBAFormat); vTex.colorSpace = THREE.SRGBColorSpace; vTex.magFilter = THREE.LinearFilter; vTex.minFilter = THREE.LinearFilter; vTex.needsUpdate = true;
  const vMat = new THREE.MeshBasicMaterial({ map: vTex, transparent: true, opacity: 1, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending });
  const vSlice = new THREE.Mesh(new THREE.PlaneGeometry(L, W), vMat); vSlice.rotation.x = -Math.PI / 2; vSlice.position.y = 0.01; vSlice.visible = false; field.add(vSlice);
  trackedTex.push(sliceTex, vTex);
  // hotspot spheres
  const hotMat = new THREE.MeshBasicMaterial({ color: 0xff3b30, transparent: true, opacity: 0.5, depthWrite: false, blending: THREE.AdditiveBlending });
  const hotMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 16, 12), hotMat, spots.length); hotMesh.frustumCulled = false; hotMesh.visible = false; field.add(hotMesh);
  const hm4 = new THREE.Matrix4(), hq = new THREE.Quaternion(), hv = new THREE.Vector3(), hs = new THREE.Vector3();

  // volume field points
  const NF = 3600, fp = new Float32Array(NF * 3), fTruth = new Float32Array(NF * 2), fCol = new Float32Array(NF * 3), fRnd = new Float32Array(NF), fPh = new Float32Array(NF);
  { const r2 = M.mulberry32(21); for (let i = 0; i < NF;) { const x = (r2() * 2 - 1) * L / 2, y = (r2() * 2 - 1) * H / 2 * 0.97, z = (r2() * 2 - 1) * W / 2 * 0.97; if (M.insideCell(x, z, 0.03)) continue; fp.set([x, y, z], i * 3); fTruth[i * 2] = M.fieldValue01(x, y, z, 'speed'); fTruth[i * 2 + 1] = M.fieldValue01(x, y, z, 'pressure'); fRnd[i] = r2(); fPh[i] = r2() * 6.28; i++; } }
  const fGeo = new THREE.BufferGeometry(); fGeo.setAttribute('position', new THREE.BufferAttribute(fp, 3)); fGeo.setAttribute('aCol', new THREE.BufferAttribute(fCol, 3));
  const fatMat = new THREE.ShaderMaterial({
    uniforms: { uSize: { value: 0.07 }, uView: { value: 600 }, uAlpha: { value: 0 } }, transparent: true, depthWrite: false,
    vertexShader: 'attribute vec3 aCol; uniform float uSize,uView; varying vec3 vC; void main(){ vC=aCol; vec4 mv=modelViewMatrix*vec4(position,1.0); gl_Position=projectionMatrix*mv; gl_PointSize=max(1.5,uSize*uView/-mv.z); }',
    fragmentShader: 'uniform float uAlpha; varying vec3 vC; void main(){ float d=length(gl_PointCoord-0.5); if(d>0.5) discard; gl_FragColor=vec4(vC, smoothstep(0.5,0.25,d)*uAlpha); }',
  });
  const fPts = new THREE.Points(fGeo, fatMat); fPts.frustumCulled = false; field.add(fPts);

  // colour legend under the output copy
  const lcv = document.createElement('canvas'); lcv.width = 256; lcv.height = 8; { const g2 = lcv.getContext('2d'); for (let i = 0; i < 256; i++) { const c = M.viridis(i / 255); g2.fillStyle = `rgb(${c[0] * 255 | 0},${c[1] * 255 | 0},${c[2] * 255 | 0})`; g2.fillRect(i, 0, 1, 8); } }
  const legTex = new THREE.CanvasTexture(lcv); legTex.colorSpace = THREE.SRGBColorSpace; trackedTex.push(legTex);
  const legend = new THREE.Group(); legend.position.set(0, -H / 2 - 1.0, 0);
  const legBar = new THREE.Mesh(new THREE.PlaneGeometry(3.6, 0.2), new THREE.MeshBasicMaterial({ map: legTex, transparent: true })); legBar.material.userData.baseOpacity = 1;
  const legTitle = label('|u| velocity magnitude (normalised)', 0.3); legTitle.position.set(0, 0.42, 0);
  const legLo = label('low', 0.26, { color: '#aeb8cc' }), legHi = label('high', 0.26, { color: '#aeb8cc' }); legLo.position.set(-2.1, 0, 0); legHi.position.set(2.1, 0, 0);
  const legNote = label('illustrative field, not a simulation result', 0.24, { color: '#f2c14e' }); legNote.position.set(0, -0.42, 0);
  legend.add(legBar, legTitle, legLo, legHi, legNote); legend.visible = false; field.add(legend);

  // ---------- INPUT CLOUD (stage 2) ----------
  const S = M.generateSamples(M.MAX_POINTS, 7), MAXN = M.MAX_POINTS;
  const wpos = new Float32Array(MAXN * 3), wtgt = new Float32Array(MAXN * 3), col = new Float32Array(MAXN * 3), grp = new Float32Array(MAXN);
  const G = LAYOUT.geomCenter, C = LAYOUT.cloudCenter, CH = LAYOUT.cloudHalf;
  for (let i = 0; i < MAXN; i++) {
    for (let a = 0; a < 3; a++) { wpos[i * 3 + a] = S.pos[i * 3 + a] + G[a]; wtgt[i * 3 + a] = S.tgt[i * 3 + a] * CH + C[a]; }
    const k = S.kind[i], c = hex(M.KIND_HEX[k]); col.set([c.r, c.g, c.b], i * 3); grp[i] = k === 0 ? 0 : 1;
  }
  // colours are display-space (custom shader): convert from THREE's linear working space back to sRGB values
  for (let k = 0; k < 5; k++) { const c = hex(M.KIND_HEX[k]).convertLinearToSRGB(); for (let i = 0; i < MAXN; i++) if (S.kind[i] === k) col.set([c.r, c.g, c.b], i * 3); }
  const cGeo = new THREE.BufferGeometry();
  cGeo.setAttribute('position', new THREE.BufferAttribute(wpos, 3)); cGeo.setAttribute('aTarget', new THREE.BufferAttribute(wtgt, 3)); cGeo.setAttribute('aCol', new THREE.BufferAttribute(col, 3));
  cGeo.setAttribute('aRand', new THREE.BufferAttribute(S.rand, 1)); cGeo.setAttribute('aGroup', new THREE.BufferAttribute(grp, 1)); cGeo.setAttribute('aJit', new THREE.BufferAttribute(S.jit, 3));
  const cMat = new THREE.ShaderMaterial({
    uniforms: { uLift: { value: 0 }, uRevI: { value: 0 }, uRevB: { value: 0 }, uSize: { value: 0.09 }, uView: { value: 600 }, uConsume: { value: 0 } }, transparent: true, depthWrite: false,
    vertexShader: `attribute vec3 aTarget; attribute vec3 aCol; attribute float aRand; attribute float aGroup; attribute vec3 aJit;
uniform float uLift,uRevI,uRevB,uSize,uView,uConsume; varying vec3 vC; varying float vA;
void main(){
  float rev = aGroup < 0.5 ? uRevI : uRevB;
  float g = smoothstep(aRand*0.6, aRand*0.6+0.4, rev);
  float l = smoothstep(aRand*0.4, aRand*0.4+0.6, uLift);
  vec3 p = mix(position + aJit*(1.0-g)*0.8, aTarget, l);
  p.y += sin(3.14159*l)*(0.5+1.5*aRand);
  float cons = smoothstep(aRand, aRand+0.12, uConsume);
  vec4 mv = modelViewMatrix*vec4(p,1.0); gl_Position = projectionMatrix*mv;
  float sz = uSize*mix(1.0,0.5,l)*(aGroup>0.5?1.2:1.0);
  gl_PointSize = max(g*(1.0-cons)*1.8, sz*uView*g*(1.0-cons)/-mv.z);
  vC = aCol*(0.9+0.25*(1.0-cons)); vA = g*(1.0-0.85*cons);
}`,
    fragmentShader: 'varying vec3 vC; varying float vA; void main(){ float d=length(gl_PointCoord-0.5); if(d>0.5||vA<0.01) discard; float core=1.0-d*1.6; gl_FragColor=vec4(vC+0.18*core, smoothstep(0.5,0.3,d)*vA); }',
  });
  const cloud = new THREE.Points(cGeo, cMat); cloud.frustumCulled = false; cloud.visible = false; cloud.renderOrder = 2; group.add(cloud);
  cGeo.setDrawRange(0, 4096);
  // cloud frame + labels + legend
  const frame = new THREE.Group(); frame.position.set(...C);
  const fm = new THREE.LineBasicMaterial({ color: 0x8aa0c0, transparent: true, opacity: 0.55 }); fm.userData.baseOpacity = 0.55;
  frame.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(CH * 2, CH * 2, CH * 2)), fm));
  const tCloud = label('Input cloud (x, y, z)', 0.34); tCloud.position.set(0, CH + 0.45, 0);
  const tCount = label('4,096 points', 0.3, { color: '#9fd3ff' }); tCount.position.set(0, -CH - 0.4, CH);
  const toGrp = new THREE.Group(); toGrp.position.set(C[0] + CH + 0.15, 0, 0);
  const tTo = label('to network →', 0.3, { color: '#f2c14e' }); tTo.center.set(0, 0.5); toGrp.add(tTo); toGrp.visible = false; group.add(toGrp);
  const tCloudG = new THREE.Group(); tCloudG.position.copy(frame.position); tCloudG.add(tCloud); group.add(tCloudG);
  frame.add(tCount); frame.visible = false; group.add(frame);
  const legendIn = new THREE.Group(); legendIn.position.set(G[0], -H / 2 - 0.55, W / 2 + 0.6);
  const kn = ['interior collocation', 'wall boundary', 'inlet boundary', 'outlet boundary', 'obstacle surface'];
  { const sp = kn.map((t, i) => label('\u25CF ' + t, 0.27, { color: '#' + M.KIND_HEX[i].toString(16).padStart(6, '0') })), rows = [[0, 1, 2], [3, 4]];
    rows.forEach((r, ri) => { const tot = r.reduce((a, k) => a + sp[k].scale.x, 0) + 0.25 * (r.length - 1); let x = -tot / 2; for (const k of r) { sp[k].center.set(0, 0.5); sp[k].position.set(x, -ri * 0.42, 0); x += sp[k].scale.x + 0.25; legendIn.add(sp[k]); } }); }
  legendIn.visible = false; group.add(legendIn);

  // ---------- state ----------
  let n = 4096, extT = null, extV = 0, extConsume = 0, channel = 'speed', autoT = 0, lift = 0, liftHold = 0, inited = false, lastTick = -1, fieldKey = '', hotKey = '', lastFlick = 0, labW = 1;
  const api = {
    LAYOUT,
    /** 0..1: untrained/noisy -> resolved. Once called, overrides the automatic ramp. Pass null to return to auto. */
    setOutputProgress(t) { extT = t === null ? null : util.clamp(t); },
    /** 0..1 strength of PDE-violation hotspots (shown only once part pinn.residualfield is revealed). */
    setPhysicsViolation(v) { extV = util.clamp(v); },
    /** 'speed' | 'pressure' */
    setOutputChannel(c) { channel = c === 'pressure' ? 'pressure' : 'speed'; fieldKey = ''; setLabel2(); },
    /** 0..1 fraction of input points already consumed by the network (they dim/shrink); for the forward-pass pulse. */
    setInputConsumed(f) { extConsume = util.clamp(f); },
    /** Live view of the displayed input cloud in WORLD coordinates (normalised cube). Arrays are views: do not mutate. */
    getInputPoints() { return { count: n, positions: wtgt.subarray(0, n * 3), colors: col.subarray(0, n * 3), kinds: S.kind.subarray(0, n), rand: S.rand.subarray(0, n), center: new THREE.Vector3(...C), half: CH, port: new THREE.Vector3(...LAYOUT.inputPort) }; },
    getAnchors() { return { inputPort: new THREE.Vector3(...LAYOUT.inputPort), outputPort: new THREE.Vector3(...LAYOUT.outputPort), outCenter: new THREE.Vector3(...LAYOUT.outCenter), maxPoints: MAXN }; },
    get pointCount() { return n; },
    get outputProgress() { return curT(); },
  };
  const curT = () => (extT !== null ? extT : autoT);
  function setLabel2() { util.setLabel(legTitle, channel === 'pressure' ? 'p pressure (normalised)' : '|u| velocity magnitude (normalised)'); }

  const setN = () => {
    n = Math.min(M.MAX_POINTS, Math.max(16, Math.round(knobs.get('batch') / 1))); cGeo.setDrawRange(0, n);
    const sz = 0.1 / Math.pow(Math.max(n, 1024) / 4096, 0.22); cMat.uniforms.uSize.value = sz;
    util.setLabel(tCount, `${n.toLocaleString('en')} points${knobs.get('batch') > M.MAX_POINTS ? ' shown (capped from ' + Math.round(knobs.get('batch')).toLocaleString('en') + ')' : ''}`);
  };
  knobs.on('batch', () => { if (!state0.disposed) setN(); }); setN();

  function paintField(t, time) {
    const e = util.smoothstep(0.05, 1, t), nz = Math.pow(1 - e, 1.4), ch = channel === 'pressure' ? 1 : 0;
    // slice
    for (let k = 0; k < TW * TH; k++) {
      const x = (k % TW) / TW, y = (k / TW | 0) / TH;
      const smooth = 0.5 + 0.28 * Math.sin(7 * x + 4 * y + time * 1.1) * Math.cos(5 * y - 3 * x + time * 0.8);
      const v = truth[k * 2 + ch] * e + (0.5 * smooth + 0.35 * rndT[k]) * (1 - e) + 0.5 * nz * (rndT[k] - 0.5) * 0.3;
      M.viridis(Math.min(1, Math.max(0, v)), tmpC); const o = k * 4;
      texData[o] = tmpC[0] * 255; texData[o + 1] = tmpC[1] * 255; texData[o + 2] = tmpC[2] * 255; texData[o + 3] = mask[k] ? 0 : 255;
    }
    sliceTex.needsUpdate = true;
    for (let i = 0; i < NF; i++) {
      const sm = 0.5 + 0.28 * Math.sin(2.1 * fp[i * 3] + 3 * fp[i * 3 + 2] + fp[i * 3 + 1] * 2.5 + time * 1.1 + fPh[i]);
      const v = fTruth[i * 2 + ch] * e + (0.5 * sm + 0.35 * fRnd[i]) * (1 - e);
      M.viridis(Math.min(1, Math.max(0, v)), tmpC); fCol[i * 3] = tmpC[0]; fCol[i * 3 + 1] = tmpC[1]; fCol[i * 3 + 2] = tmpC[2];
    }
    fGeo.attributes.aCol.needsUpdate = true;
  }
  function paintViolation(v, time) {
    for (let k = 0; k < TW * TH; k++) { const a = hot[k] * v * 255, o = k * 4; vData[o] = a; vData[o + 1] = a * 0.12; vData[o + 2] = a * 0.1; vData[o + 3] = 255; }
    vTex.needsUpdate = true;
  }
  function placeHot(v, time, rev) {
    for (let i = 0; i < spots.length; i++) {
      const h = spots[i], pulse = 1 + 0.18 * Math.sin(time * 4 + i * 1.7), s = (0.07 + 0.1 * h.s) * v * pulse * rev;
      hm4.compose(hv.set(h.x, h.y, h.z), hq, hs.setScalar(Math.max(s, 1e-4))); hotMesh.setMatrixAt(i, hm4);
    }
    hotMesh.instanceMatrix.needsUpdate = true;
  }

  function tick(dt, state) {
    if (lastTick === state.time) return; lastTick = state.time;
    const rv = (id) => state.reveal[id] ?? 0, still = state.reducedMotion;
    // lift target & snap-on-first-frame (deep link / reduced motion)
    const rI = rv('sampling.interior'), rB = rv('sampling.boundary');
    const target = Math.min(rI, rB) > 0.92 ? 1 : 0;
    if (!inited) { lift = target; inited = true; }
    else if (still) lift = target;
    else if (target === 1) { liftHold += dt; if (liftHold > 1.1) lift = Math.min(1, lift + dt / 2.6); }
    else { liftHold = 0; lift = Math.max(0, lift - dt / 1.2); }
    // gate parts
    util.gateGroup(duct, rv('geometry.duct')); util.gateGroup(slot, rv('geometry.fieldslot'));
    const rf = rv('forward.field'), rvio = rv('pinn.residualfield');
    cMat.uniforms.uRevI.value = rI; cMat.uniforms.uRevB.value = rB; cMat.uniforms.uLift.value = util.smoothstep(0, 1, lift); cMat.uniforms.uConsume.value = extConsume;
    cloud.visible = rI > 0.001 || rB > 0.001;
    const rc = Math.max(rI, rB); labW += ((state.stageIndex <= 1 ? 1 : 0) - labW) * Math.min(1, dt * 4); if (still) labW = state.stageIndex <= 1 ? 1 : 0;
    util.gateGroup(frame, rc * util.smoothstep(0.1, 0.9, lift)); util.gateGroup(legendIn, rc * labW); util.gateGroup(tCloudG, rc * labW * util.smoothstep(0.1, 0.9, lift));
    // legend only while in the sampling stage region visible; fade the end-arrow label unless this is the sampling stage
    util.gateGroup(toGrp, state.stageId === 'sampling' ? rc * util.smoothstep(0.6, 1, lift) : 0);
    // arrows bob
    const ph = still ? 0 : (state.time * 0.9) % 1;
    inA.position.x = inA.userData.x0 + 0.25 * Math.sin(ph * 6.2832) * (still ? 0 : 1); outA.position.x = outA.userData.x0 + 0.25 * Math.sin(ph * 6.2832) * (still ? 0 : 1);
    const px = (innerHeight * Math.min(devicePixelRatio || 1, 2)) / 2 * ctx.camera.projectionMatrix.elements[5];
    cMat.uniforms.uView.value = px; fatMat.uniforms.uView.value = px;
    // output field
    if (extT === null) { if (rf > 0.5) autoT = still ? 1 : Math.min(1, autoT + dt / 3.2); else if (rf < 0.01) autoT = 0; }
    const t = curT(), vis = rf > 0.001;
    field.visible = vis;
    if (vis) {
      sliceMat.opacity = 0.96 * rf; fatMat.uniforms.uAlpha.value = 0.85 * rf; legend.visible = true;
      util.gateGroup(legend, rf);
      const key = t.toFixed(3) + channel, animating = t < 0.995 && !still;
      if (key !== fieldKey || (animating && state.time - lastFlick > 0.066)) { fieldKey = key; lastFlick = state.time; paintField(t, still ? 0 : state.time); }
      util.setLabel(tSlot, rf > 0.5 ? 'Network output: predicted field (illustrative)' : 'Output field slot (empty)');
    }
    const vv = vis ? extV * rvio : 0;
    vSlice.visible = vv > 0.002; hotMesh.visible = vv > 0.002;
    if (vSlice.visible) { const hk = vv.toFixed(3); if (hk !== hotKey) { hotKey = hk; paintViolation(vv, state.time); } placeHot(vv, state.time, 1); vMat.opacity = 1; }
    // readouts
    if (state.stageId === 'sampling') {
      const c = M.countKinds(S.kind, n);
      hud.readout('pts', 'Input points (x,y,z)', n.toLocaleString('en') + (knobs.get('batch') > M.MAX_POINTS ? ' (display cap)' : ''));
      hud.readout('int', 'Interior collocation', c[0].toLocaleString('en'));
      hud.readout('bnd', 'Boundary (wall/in/out/cell)', `${c[1]} / ${c[2]} / ${c[3]} / ${c[4]}`);
    }
  }

  const camFor = (cx, halfW, minD, ty = 0) => {
    const aspect = innerWidth / innerHeight, tanH = Math.tan(25 * Math.PI / 180);
    const d = Math.max(minD, halfW / (tanH * Math.min(aspect, 2.2)), 3.4 / tanH);
    return { pos: [cx - 0.04 * d, 0.36 * d + ty, 0.93 * d], target: [cx, ty, 0] };
  };
  const base = { update: tick, enter() {}, exit() {} };
  const stages = {
    geometry: { ...base,
      get camera() { return innerWidth / innerHeight < 1 ? camFor(G[0], 5.6, 8) : camFor(0.9, 15.8, 9); } },
    sampling: { ...base,
      get camera() { return camFor(-6.4, innerWidth / innerHeight < 1 ? 7.4 : 6.3, 12, -0.9); } },
  };
  setLabel2();
  return { group, stages, api, dispose() { state0.disposed = true; util.disposeObject(group); trackedTex.forEach((t) => t.dispose()); } };
}
