// graph.js - autograd graphs as the memory/compute carrier.
//   fillGraph(out, cfg, variant)  PURE (typed arrays only, node-testable): lays out forward + backward nodes and edges.
//     Node count == model.autogradGraph(cfg, variant).totalNodes. Node SIZE ~ cbrt(saved-intermediate bytes) (memory).
//     Node/edge GLOW ~ sqrt(FLOPs of the op) (compute). Estimates come from model.js; nothing here is measured.
//   createGraph(THREE, variant, origin) builds the GPU objects once (fixed capacity, instanced) and uploads on rebuild().
import { layerDims, linearLayers, towerFactor, autogradGraph, memoryBreakdown, resRatio } from './model.js';

export const GR = { x0: -8, x1: 3.6, cap: 3600, edgeCap: 9600 };
const REF_BYTES = 4096 * 128 * 4, REF_FLOPS = 2 * 4096 * 128 * 128;
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const hash = (n) => { const s = Math.sin(n * 127.1 + 311.7) * 43758.5453; return s - Math.floor(s); };

export function makeBuffers(nodeCap = GR.cap, edgeCap = GR.edgeCap) {
  return { nodeCap, edgeCap, nodes: { pos: new Float32Array(nodeCap * 3), A: new Float32Array(nodeCap * 4), B: new Float32Array(nodeCap * 2) },
    edges: { pos: new Float32Array(edgeCap * 6), A: new Float32Array(edgeCap * 8), S: new Float32Array(edgeCap * 4) }, nNodes: 0, nEdges: 0 };
}

/** variant 'supervised' (data-loss graph) or 'physics'. yFwd/yBwd = centre heights of the forward / backward slab. */
export function fillGraph(out, cfg, variant, yFwd, yBwd) {
  const L = linearLayers(cfg), F = 3 * L, dims = layerDims(cfg), mem = memoryBreakdown(cfg), rho = resRatio(cfg);
  const phys = variant === 'physics';
  const T = phys ? towerFactor(cfg) : 1, R = phys ? 6 * cfg.nResiduals : 0, fwdN = F * T + R;
  const expect = autogradGraph(cfg, variant);
  if (2 * fwdN !== expect.totalNodes) throw new Error('graph layout/model mismatch');
  if (2 * fwdN > out.nodeCap) throw new Error('graph capacity');
  const GY = phys ? Math.ceil(Math.sqrt(T)) : 1, GZ = phys ? Math.ceil(T / GY) : 1;
  const dy = Math.min(0.46, 2.3 / GY), dz = Math.min(0.85, 2.5 / GZ);
  const bytesPer = mem.perLayerActivations; // per boundary, batch rows
  const N = out.nodes, E = out.edges; let ne = 0;
  const colX = (c) => GR.x0 + (c / (F - 1)) * (GR.x1 - GR.x0);
  const nodePos = (b, c, bwd) => {
    const yc = bwd ? yBwd : yFwd;
    if (b < T) { const gy = b % GY, gz = Math.floor(b / GY); return [colX(c), yc - (gy - (GY - 1) / 2) * dy, ((GZ - 1) / 2 - gz) * dz]; }
    const k = b - T; // residual-assembly nodes: a small stack right of the last column
    return [GR.x1 + 0.9 + (k % 2) * 0.35, yc + ((Math.floor(k / 2) - (R / 4 - 0.5)) * 0.2), ((k % 3) - 1) * 0.3];
  };
  const idx = (b, c, bwd) => (bwd ? fwdN : 0) + (b < T ? b * F + c : F * T + (b - T));
  const rankOf = (b, seed) => (phys ? clamp((1 - b / Math.max(T, 2)) * 0.55 + seed * 0.3, 0, 0.9) : 0);
  const put = (bwd, b, c) => {
    const i = idx(b, c, bwd), deriv = phys && b > 0, res = b >= T;
    const l = res ? L - 1 : Math.min(Math.floor(c / 3), L - 1), j = res ? 0 : c % 3;
    const rows = deriv ? rho : 1, bytes = bytesPer[l + 1] * rows * (res ? 0.5 : 1);
    const flopsN = j === 0 && !res ? 2 * cfg.batch * rows * dims[l] * dims[l + 1] : res ? REF_FLOPS * 0.05 : REF_FLOPS * 0.01;
    const size = clamp(0.085 * Math.cbrt(bytes / REF_BYTES), 0.02, 0.26) * (bwd ? 0.8 : 1);
    const glow = clamp(Math.sqrt(flopsN / REF_FLOPS), 0.12, 1.3);
    const p = nodePos(b, c, bwd), seed = hash(i + 1);
    const tt = res ? (bwd ? 0 : 1) : (bwd ? 1 - c / (F - 1) : c / (F - 1)) * 0.9 + 0.1 * (b / Math.max(T, 2));
    N.pos[i * 3] = p[0]; N.pos[i * 3 + 1] = p[1]; N.pos[i * 3 + 2] = p[2];
    N.A[i * 4] = size; N.A[i * 4 + 1] = glow; N.A[i * 4 + 2] = tt; N.A[i * 4 + 3] = (bwd ? 2 : 0) + (deriv || res ? 1 : 0);
    N.B[i * 2] = rankOf(b, seed); N.B[i * 2 + 1] = seed;
  };
  for (let b = 0; b < T; b++) for (let c = 0; c < F; c++) { put(false, b, c); put(true, b, c); }
  for (let k = 0; k < R; k++) { put(false, T + k, 0); put(true, T + k, 0); }
  const edge = (i, k, etype) => {
    if (ne >= out.edgeCap) return;
    for (const [v, n] of [[0, i], [1, k]]) {
      const o = ne * 2 + v;
      E.pos[o * 3] = N.pos[n * 3]; E.pos[o * 3 + 1] = N.pos[n * 3 + 1]; E.pos[o * 3 + 2] = N.pos[n * 3 + 2];
      E.A[o * 4] = N.A[n * 4 + 2]; E.A[o * 4 + 1] = N.A[n * 4 + 3]; E.A[o * 4 + 2] = N.A[n * 4 + 1]; E.A[o * 4 + 3] = N.B[n * 2];
      E.S[o * 2] = N.B[n * 2 + 1]; E.S[o * 2 + 1] = etype;
    }
    ne++;
  };
  for (const bwd of [false, true]) {
    for (let b = 0; b < T; b++) {
      for (let c = 0; c < F - 1; c++) edge(idx(b, c, bwd), idx(b, c + 1, bwd), 0);
      if (b > 0) for (let c = 0; c < F; c += 1) edge(idx(b, c, bwd), idx((b - 1) >> 1, c, bwd), 1);
    }
    for (let k = 0; k < R; k++) for (const q of [k % T, (k * 7 + 3) % T]) edge(idx(T + k, 0, bwd), idx(q, F - 1, bwd), 1);
  }
  for (let b = 0; b < T; b++) for (let c = 0; c < F; c++) edge(idx(b, c, false), idx(b, c, true), 2);
  for (let k = 0; k < R; k++) edge(idx(T + k, 0, false), idx(T + k, 0, true), 2);
  out.nNodes = 2 * fwdN; out.nEdges = ne;
  return out;
}

const COMMON = `
uniform float uU, uSage, uLambda, uAlpha; uniform vec2 uFw, uBw; uniform vec3 uLA, uLB, uLC;
float disOf(float r){ return smoothstep(r * 0.6, r * 0.6 + 0.4, uSage); }
vec3 lanePt(float t){ return mix(mix(uLA, uLC, t), mix(uLC, uLB, t), t); }
vec3 moved(vec3 p, float seed, float d){
  float e = d * d * (3.0 - 2.0 * d); vec3 q = lanePt(fract(seed * 13.7));
  vec3 m = mix(p, q, e); float k = sin(e * 3.14159);
  m.z += k * (fract(seed * 5.3) - 0.5) * 3.0; m.y += k * 0.8; return m;
}
vec3 baseCol(float kind, float lit, float glow){
  bool bwd = kind > 1.5, der = mod(kind, 2.0) > 0.5; vec3 c;
  if (!bwd) c = mix(vec3(0.23,0.72,0.58) * 0.45, vec3(0.23,0.72,0.58), 0.55 + 0.45 * lit);
  else c = mix(der ? vec3(0.34,0.24,0.38) : vec3(0.27,0.32,0.45), der ? vec3(0.98,0.42,0.36) : vec3(0.94,0.54,0.14), lit);
  if (!bwd && der) c = mix(vec3(0.88,0.25,0.54) * 0.55, vec3(0.88,0.25,0.54), 0.55 + 0.45 * lit);
  return c;
}
`;
const NODE_VS = COMMON + `
attribute vec3 iPos; attribute vec4 iA; attribute vec2 iB;
varying vec3 vC; varying float vK;
void main(){
  float kind = iA.w; bool bwd = kind > 1.5; bool der = mod(kind, 2.0) > 0.5;
  vec2 w = bwd ? uBw : uFw; float s = w.x + iA.z * (w.y - w.x);
  float lit = step(s, uU) * step(0.0001, uU), head = lit * exp(-(uU - s) * 22.0);
  float d = disOf(iB.x);
  float glow = iA.y * (der ? uLambda : 1.0);
  vec3 col = baseCol(kind, lit, glow) * (0.6 + 0.5 * glow) + head * vec3(1.0, 0.85, 0.6) * 0.75;
  col = mix(col, vec3(0.18, 0.82, 0.9), smoothstep(0.05, 0.5, d));
  col = mix(col, vec3(0.95, 0.76, 0.3), smoothstep(0.8, 1.0, d));
  float size = iA.x * (1.0 + 0.7 * head) * (1.0 - smoothstep(0.55, 1.0, d));
  vec3 p = moved(iPos, iB.y, d);
  vec3 n = normalize(normalMatrix * normal);
  vC = col * (0.55 + 0.6 * max(0.0, n.z * 0.7 + n.y * 0.3 + 0.2));
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p + position * size, 1.0);
  if (size < 0.003) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
}`;
const NODE_FS = `varying vec3 vC; uniform float uAlpha; void main(){ gl_FragColor = vec4(vC, 1.0); }`;
const EDGE_VS = COMMON + `
attribute vec4 aA; attribute vec2 aS; // aA: t, kind, glow, rank ; aS: seed, edge type
varying vec4 vC;
void main(){
  float kind = aA.y; bool bwd = kind > 1.5; bool der = mod(kind, 2.0) > 0.5;
  vec2 w = bwd ? uBw : uFw; float s = w.x + aA.x * (w.y - w.x);
  float lit = step(s, uU) * step(0.0001, uU), head = lit * exp(-(uU - s) * 22.0);
  float d = disOf(aA.w);
  float glow = aA.z * (der ? uLambda : 1.0);
  vec3 col = baseCol(kind, lit, glow) + head * vec3(1.0, 0.8, 0.5);
  float a = (aS.y > 1.5 ? 0.05 : 0.10) + 0.16 * glow * (aS.y > 1.5 ? 0.0 : 1.0) + head * 0.45;
  a *= uAlpha * (1.0 - smoothstep(0.35, 0.9, d));
  col = mix(col, vec3(0.18, 0.82, 0.9), smoothstep(0.05, 0.5, d));
  vC = vec4(col, a);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(moved(position, aS.x, d), 1.0);
}`;
const EDGE_FS = `varying vec4 vC; void main(){ gl_FragColor = vC; }`;

export function createGraph(THREE, variant, yFwd, yBwd, lane) {
  const group = new THREE.Group(), buf = makeBuffers(variant === 'physics' ? GR.cap : 120, variant === 'physics' ? GR.edgeCap : 400);
  const uni = {
    uU: { value: 0 }, uSage: { value: 0 }, uLambda: { value: 1 }, uAlpha: { value: 1 }, uFw: { value: new THREE.Vector2(0.2, 0.4) }, uBw: { value: new THREE.Vector2(0.5, 0.7) },
    uLA: { value: lane.a }, uLB: { value: lane.b }, uLC: { value: lane.c },
  };
  const base = new THREE.IcosahedronGeometry(1, 1);
  const ng = new THREE.InstancedBufferGeometry(); ng.index = base.index; ng.setAttribute('position', base.getAttribute('position')); ng.setAttribute('normal', base.getAttribute('normal'));
  const nPos = new THREE.InstancedBufferAttribute(buf.nodes.pos, 3), nA = new THREE.InstancedBufferAttribute(buf.nodes.A, 4), nB = new THREE.InstancedBufferAttribute(buf.nodes.B, 2);
  [nPos, nA, nB].forEach((a) => a.setUsage(THREE.DynamicDrawUsage));
  ng.setAttribute('iPos', nPos); ng.setAttribute('iA', nA); ng.setAttribute('iB', nB); ng.instanceCount = 0;
  const nm = new THREE.ShaderMaterial({ vertexShader: NODE_VS, fragmentShader: NODE_FS, uniforms: uni });
  const nodes = new THREE.Mesh(ng, nm); nodes.frustumCulled = false; nodes.renderOrder = 2; group.add(nodes);

  const eg = new THREE.BufferGeometry();
  const ePos = new THREE.BufferAttribute(buf.edges.pos, 3), eA = new THREE.BufferAttribute(buf.edges.A, 4), eS = new THREE.BufferAttribute(buf.edges.S, 2);
  [ePos, eA, eS].forEach((a) => a.setUsage(THREE.DynamicDrawUsage));
  eg.setAttribute('position', ePos); eg.setAttribute('aA', eA); eg.setAttribute('aS', eS); eg.setDrawRange(0, 0);
  const em = new THREE.ShaderMaterial({ vertexShader: EDGE_VS, fragmentShader: EDGE_FS, uniforms: uni, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  const edges = new THREE.LineSegments(eg, em); edges.frustumCulled = false; edges.renderOrder = 1; group.add(edges);

  return {
    group, uni, stats: buf,
    rebuild(cfg) {
      fillGraph(buf, cfg, variant, yFwd, yBwd);
      [nPos, nA, nB, ePos, eA, eS].forEach((a) => { a.needsUpdate = true; });
      ng.instanceCount = buf.nNodes; eg.setDrawRange(0, buf.nEdges * 2);
      return buf.nNodes;
    },
    dispose() { base.dispose(); ng.dispose(); nm.dispose(); eg.dispose(); em.dispose(); },
  };
}
