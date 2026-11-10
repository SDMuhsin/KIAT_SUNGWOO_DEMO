// physics_graph.js - REUSABLE autograd-graph drawing helper (instanced nodes + edges, shader-driven backward wavefront).
//   const g = createPhysicsGraph(ctx, { variant: 'physics' | 'supervised', width: 8 });
//   g.rebuild(cfg)        - (re)build from a model cfg (node count == model.autogradGraph(cfg, variant).totalNodes). No allocation on rebuild.
//   g.animate(dt, opts)   - advance the backward wavefront (hops/s, same speed for any graph) and loop it. g.setSweep(h) to scrub.
//   g.setCollapse(t)      - t=1: graph ELIMINATED (nodes/edges implode to a point and vanish). SAGE uses this. t=0 restores.
//   g.setAlpha(a), g.group (position/scale it freely), g.stats, g.dispose()
// Colours: forward nodes by derivative order (green = plain net, magenta shades = derivative branches), backward nodes grey until the
// orange wavefront reaches them. Pure graph construction lives in physics_math.js (node-tested).
import { buildGraph, graphCounts } from './physics_math.js';

const NODE_VS = `
attribute vec3 iPos; attribute float iSize; attribute vec4 iMeta; // ord, bwd, t(hops), rand
uniform float uSweep, uCollapse, uAlpha; uniform vec3 uCenter;
varying vec3 vN; varying vec3 vCol; varying float vA;
void main(){
  float c = smoothstep(iMeta.w*0.55, iMeta.w*0.55+0.45, uCollapse);
  vec3 p = mix(iPos, uCenter, c*0.9);
  float sz = iSize*(1.0-c);
  float lit = step(iMeta.z, uSweep), age = uSweep - iMeta.z;
  vec3 base;
  float o = iMeta.x;
  if (o < 0.5) base = vec3(0.23,0.72,0.58); else if (o < 1.5) base = vec3(0.88,0.25,0.54); else if (o < 2.5) base = vec3(1.0,0.48,0.69); else if (o < 3.5) base = vec3(1.0,0.69,0.82); else base = vec3(1.0,0.93,0.95);
  vec3 orange = vec3(0.94,0.54,0.14);
  if (iMeta.y > 0.5) {
    vec3 dim = vec3(0.26,0.31,0.43);
    vCol = mix(dim, orange, lit) + lit*exp(-age*0.45)*vec3(0.55,0.5,0.4);
  } else {
    vCol = base*0.8 + lit*exp(-age*0.8)*(orange*0.9);
  }
  vA = uAlpha*(1.0-c);
  vN = normalMatrix*normal;
  gl_Position = projectionMatrix*modelViewMatrix*vec4(p + position*sz, 1.0);
}`;
const NODE_FS = `
varying vec3 vN; varying vec3 vCol; varying float vA;
void main(){ float d = 0.5 + 0.5*normalize(vN).z; gl_FragColor = vec4(vCol*(0.45+0.75*d), vA); if (vA < 0.01) discard; }`;
// Edges are instanced screen-space quads (constant pixel width, additive glow) instead of 1px GL lines, so they stay visible on projectors.
const EDGE_VS = `
attribute vec3 aP0; attribute vec3 aP1; attribute vec4 aMeta; // type, bwd, t0, t1 ; rand packed below
attribute float aRand;
uniform float uSweep, uCollapse, uAlpha, uPx, uBoost; uniform vec3 uCenter; uniform vec2 uRes;
varying vec4 vC; varying float vE;
void main(){
  float c = smoothstep(aRand*0.55, aRand*0.55+0.45, uCollapse);
  vec3 p0 = mix(aP0, uCenter, c*0.9), p1 = mix(aP1, uCenter, c*0.9);
  float u = position.x, side = position.y;
  float tt = mix(aMeta.z, aMeta.w, u);
  float lit = step(tt, uSweep), age = uSweep - tt;
  float ty = aMeta.x;
  vec3 col; float a;
  if (aMeta.y > 0.5) {
    if (ty > 2.5) { col = vec3(0.62,0.55,0.3); a = 0.2; } else { col = vec3(0.34,0.42,0.6); a = 0.36; }
    vec3 hot = vec3(1.0,0.6,0.18);
    col = mix(col, hot, lit); a = mix(a, 0.9, lit) + lit*exp(-age*0.5)*0.35;
  } else {
    if (ty > 0.5 && ty < 2.5) { col = vec3(1.0,0.32,0.62); a = 0.62; } else { col = vec3(0.4,0.6,0.9); a = 0.45; }
    col += lit*exp(-age*0.8)*vec3(0.5,0.3,0.0);
  }
  vC = vec4(col, min(1.0, a*uBoost)*uAlpha*(1.0-c)); vE = side;
  vec4 c0 = projectionMatrix*modelViewMatrix*vec4(p0,1.0), c1 = projectionMatrix*modelViewMatrix*vec4(p1,1.0);
  vec4 cp = mix(c0, c1, u);
  vec2 s0 = c0.xy/c0.w*uRes, s1 = c1.xy/c1.w*uRes;
  vec2 d = s1 - s0; float L = length(d); d = L > 1e-4 ? d/L : vec2(1.0,0.0);
  vec2 nrm = vec2(-d.y, d.x);
  cp.xy += nrm*side*uPx*cp.w/uRes;
  gl_Position = cp;
}`;
const EDGE_FS = `varying vec4 vC; varying float vE; void main(){ if (vC.a < 0.01) discard; float f = 1.0 - 0.55*abs(vE); gl_FragColor = vec4(vC.rgb, vC.a*f); }`;

export function createPhysicsGraph(ctx, { variant = 'physics', width = 8, nodeSize = 0.085, edgePx = 1.0, edgeBoost = 1.0 } = {}) {
  const { THREE } = ctx;
  const worst = { depth: 12, pdeOrder: 3, nResiduals: 8, inDim: 3 };
  const cap = graphCounts(worst, variant), CN = cap.nodes + 8, CE = cap.edges + 8;
  const group = new THREE.Group(); group.name = 'physics-graph-' + variant;
  const uni = { uSweep: { value: -1 }, uCollapse: { value: 0 }, uAlpha: { value: 1 }, uCenter: { value: new THREE.Vector3() }, uPx: { value: edgePx }, uBoost: { value: edgeBoost }, uRes: { value: new THREE.Vector2(1280, 720) } };
  // nodes
  const sph = new THREE.SphereGeometry(1, 8, 6), ng = new THREE.InstancedBufferGeometry();
  ng.index = sph.index; ng.setAttribute('position', sph.getAttribute('position')); ng.setAttribute('normal', sph.getAttribute('normal'));
  const iPos = new THREE.InstancedBufferAttribute(new Float32Array(CN * 3), 3), iSize = new THREE.InstancedBufferAttribute(new Float32Array(CN), 1), iMeta = new THREE.InstancedBufferAttribute(new Float32Array(CN * 4), 4);
  ng.setAttribute('iPos', iPos); ng.setAttribute('iSize', iSize); ng.setAttribute('iMeta', iMeta); ng.instanceCount = 0;
  const nodeMat = new THREE.ShaderMaterial({ vertexShader: NODE_VS, fragmentShader: NODE_FS, uniforms: uni, transparent: true });
  const nodes = new THREE.Mesh(ng, nodeMat); nodes.frustumCulled = false; nodes.renderOrder = 3;
  // edges: one instanced quad per edge
  const eg = new THREE.InstancedBufferGeometry();
  eg.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, -1, 0, 1, -1, 0, 0, 1, 0, 1, 1, 0]), 3)); eg.index = new THREE.BufferAttribute(new Uint16Array([0, 1, 2, 2, 1, 3]), 1);
  const eP0 = new THREE.InstancedBufferAttribute(new Float32Array(CE * 3), 3), eP1 = new THREE.InstancedBufferAttribute(new Float32Array(CE * 3), 3), eMeta = new THREE.InstancedBufferAttribute(new Float32Array(CE * 4), 4), eRand = new THREE.InstancedBufferAttribute(new Float32Array(CE), 1);
  eg.setAttribute('aP0', eP0); eg.setAttribute('aP1', eP1); eg.setAttribute('aMeta', eMeta); eg.setAttribute('aRand', eRand); eg.instanceCount = 0;
  const edgeMat = new THREE.ShaderMaterial({ vertexShader: EDGE_VS, fragmentShader: EDGE_FS, uniforms: uni, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide });
  const edges = new THREE.Mesh(eg, edgeMat); edges.frustumCulled = false; edges.renderOrder = 2;
  const _sz = new THREE.Vector2();
  edges.onBeforeRender = (renderer) => { renderer.getDrawingBufferSize(_sz); uni.uRes.value.copy(_sz).multiplyScalar(0.5); uni.uPx.value = S.px * _sz.y / Math.max(1, innerHeight) ; };
  group.add(edges, nodes);

  const S = { px: edgePx, sweep: -1, hold: 0, built: null, collapse: 0, alpha: 1, disposed: false };
  const stats = { variant, nodes: 0, edges: 0, fwdNodes: 0, hops: 0, T: 1, L: 1, bounds: null, version: 0 };

  function rebuild(cfg) {
    const g = buildGraph(cfg, variant, { width }); S.built = g;
    const n = g.nodeCount, nd = g.nodes, sz = Math.min(nodeSize, g.sx * 0.38);
    const p = iPos.array, m = iMeta.array, s = iSize.array;
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < n; i++) {
      p[i * 3] = nd.x[i]; p[i * 3 + 1] = nd.y[i]; p[i * 3 + 2] = nd.z[i]; s[i] = sz * (nd.ord[i] === 4 ? 1.25 : 1);
      const r = ((i * 2654435761) >>> 0) / 4294967296;
      m[i * 4] = nd.ord[i]; m[i * 4 + 1] = nd.bwd[i]; m[i * 4 + 2] = nd.t[i]; m[i * 4 + 3] = r;
      cx += nd.x[i]; cy += nd.y[i]; cz += nd.z[i];
    }
    uni.uCenter.value.set(cx / n, cy / n, cz / n);
    ng.instanceCount = n; iPos.needsUpdate = iSize.needsUpdate = iMeta.needsUpdate = true;
    const E = g.edges, q0 = eP0.array, q1 = eP1.array, em = eMeta.array, er = eRand.array;
    for (let e = 0; e < E.count; e++) {
      const a = E.a[e], b = E.b[e];
      q0[e * 3] = nd.x[a]; q0[e * 3 + 1] = nd.y[a]; q0[e * 3 + 2] = nd.z[a]; q1[e * 3] = nd.x[b]; q1[e * 3 + 1] = nd.y[b]; q1[e * 3 + 2] = nd.z[b];
      em[e * 4] = E.type[e]; em[e * 4 + 1] = E.bwd[e]; em[e * 4 + 2] = nd.t[a]; em[e * 4 + 3] = nd.t[b]; er[e] = ((e * 40503 + 7) % 1000) / 1000;
    }
    eg.instanceCount = E.count; eP0.needsUpdate = eP1.needsUpdate = eMeta.needsUpdate = eRand.needsUpdate = true;
    Object.assign(stats, { nodes: n, edges: E.count, fwdNodes: g.fwdN, hops: g.hops, T: g.T, L: g.L, bounds: g.bounds, version: stats.version + 1 });
    return stats;
  }
  /** Advance the wavefront. opts: {speed hops/s (default 6), pause seconds at the end (default 1.4), loop (default true)}. */
  function animate(dt, { speed = 6, pause = 1.4, loop = true } = {}) {
    if (!S.built) return;
    if (S.sweep < S.built.hops + 6) S.sweep += dt * speed;
    else if (loop) { S.hold += dt; if (S.hold > pause) { S.hold = 0; S.sweep = -2; } }
    uni.uSweep.value = S.sweep;
  }
  const api = {
    group, stats, rebuild, animate, variant,
    /** Edge pixel width (at 720p-equivalent) and brightness boost; stage 10 uses thick glowing edges. */
    setEdgeStyle(px, boost) { S.px = px; uni.uBoost.value = boost; },
    setSweep(h) { S.sweep = h; uni.uSweep.value = h; },
    get sweep() { return S.sweep; }, get maxHops() { return S.built?.hops ?? 0; }, get done() { return S.built ? S.sweep >= S.built.hops : false; },
    /** t=1 means the graph is eliminated (SAGE); animates nothing itself - drive it from your own timeline. */
    setCollapse(t) { S.collapse = Math.min(1, Math.max(0, t)); uni.uCollapse.value = S.collapse; group.visible = S.collapse < 0.999 && S.alpha > 0.002; },
    get collapse() { return S.collapse; },
    setAlpha(a) { S.alpha = a; uni.uAlpha.value = a; group.visible = a > 0.002 && S.collapse < 0.999; },
    get built() { return S.built; },
    dispose() { S.disposed = true; ng.dispose();  sph.dispose(); eg.dispose(); nodeMat.dispose(); edgeMat.dispose(); group.removeFromParent(); },
  };
  group.visible = false;
  return api;
}

/** Instanced thin bars between two points (one draw call): bars.set(i, a, b, thickness, color) ... bars.commit(count). */
export function createBars(THREE, cap, opacity = 0.9) {
  const geo = new THREE.BoxGeometry(1, 1, 1), mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity });
  mat.userData.baseOpacity = opacity;
  const mesh = new THREE.InstancedMesh(geo, mat, cap); mesh.frustumCulled = false; mesh.count = 0;
  mesh.setColorAt(0, new THREE.Color(1, 1, 1));
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), v = new THREE.Vector3(), s = new THREE.Vector3(), d = new THREE.Vector3(), col = new THREE.Color(), X = new THREE.Vector3(1, 0, 0);
  return {
    mesh,
    set(i, a, b, th, color) {
      d.set(b.x - a.x, b.y - a.y, b.z - a.z); const len = Math.max(1e-4, d.length()); d.multiplyScalar(1 / len);
      q.setFromUnitVectors(X, d); v.set((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2); s.set(len, th, th);
      mesh.setMatrixAt(i, m4.compose(v, q, s)); mesh.setColorAt(i, col.set(color));
    },
    commit(count) { mesh.count = count; mesh.instanceMatrix.needsUpdate = true; if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true; },
    dispose() { geo.dispose(); mat.dispose(); mesh.dispose(); },
  };
}
export default createPhysicsGraph;
