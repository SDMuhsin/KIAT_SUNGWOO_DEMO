// net.js - the neural network: instanced neurons (green) + weight links (blue), laid out as columns left to right.
// rebuild() only rewrites fixed-capacity typed arrays (no allocation); waves (forward / backward / optimizer) are three uniforms.
export const NET = { x0: -8, x1: 2, y: 0.55, maxCols: 14, maxN: 64, spacing: 0.27, K: 4 };
export const hiddenCount = (width) => Math.max(6, Math.min(NET.maxN, Math.round(Math.pow(width, 0.62))));

const NVS = `
attribute vec3 iPos; attribute float iL;
uniform float uF, uB, uO, uAdam, uPulse;
varying vec3 vC;
void main(){
  float fl = smoothstep(iL, iL + 0.04, uF) * step(0.001, uF);
  float fh = exp(-abs(uF - iL) * 9.0) * step(0.001, uF) * step(uF, 1.04);
  float bl = smoothstep(1.0 - iL, 1.0 - iL + 0.04, uB) * step(0.001, uB);
  float bh = exp(-abs(uB - (1.0 - iL)) * 9.0) * step(0.001, uB) * step(uB, 1.04);
  vec3 col = vec3(0.23, 0.72, 0.58) * (0.5 + 0.25 * fl) + vec3(0.6, 0.95, 0.8) * fh * 0.8;
  col = mix(col, vec3(0.94, 0.54, 0.14) * 0.9, bl * 0.55) + vec3(1.0, 0.7, 0.3) * bh * 0.6;
  col += mix(vec3(0.95, 0.9, 0.8), vec3(0.62, 0.42, 0.84), uAdam) * uO * 0.5;
  float s = 1.0 + 0.7 * (fh + bh) + 0.25 * uO;
  vC = col * (0.55 + 0.45 * max(0.0, normalize(normalMatrix * normal).z + 0.4));
  gl_Position = projectionMatrix * modelViewMatrix * vec4(iPos + position * 0.085 * s, 1.0);
}`;
const NFS = `varying vec3 vC; void main(){ gl_FragColor = vec4(vC, 1.0); }`;
const EVS = `
attribute float aL;
uniform float uF, uB, uO, uAdam;
varying vec4 vC;
void main(){
  float fh = exp(-abs(uF - aL) * 9.0) * step(0.001, uF) * step(uF, 1.04);
  float bh = exp(-abs(uB - (1.0 - aL)) * 9.0) * step(0.001, uB) * step(uB, 1.04);
  vec3 col = vec3(0.29, 0.56, 0.89) * 0.75 + vec3(0.4, 0.9, 0.7) * fh + vec3(1.0, 0.6, 0.2) * bh;
  col = mix(col, mix(vec3(0.95, 0.9, 0.8), vec3(0.65, 0.45, 0.9), uAdam), uO * 0.8);
  vC = vec4(col, 0.2 + 0.5 * (fh + bh) + 0.35 * uO);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
const EFS = `varying vec4 vC; void main(){ gl_FragColor = vC; }`;

export function createNet(THREE) {
  const group = new THREE.Group();
  const maxNeurons = NET.maxCols * NET.maxN;
  const uni = { uF: { value: 0 }, uB: { value: 0 }, uO: { value: 0 }, uAdam: { value: 1 } };

  const base = new THREE.IcosahedronGeometry(1, 1);
  const ng = new THREE.InstancedBufferGeometry();
  ng.index = base.index; ng.setAttribute('position', base.getAttribute('position')); ng.setAttribute('normal', base.getAttribute('normal'));
  const iPos = new Float32Array(maxNeurons * 3), iL = new Float32Array(maxNeurons);
  const aPos = new THREE.InstancedBufferAttribute(iPos, 3), aL = new THREE.InstancedBufferAttribute(iL, 1);
  aPos.setUsage(THREE.DynamicDrawUsage); aL.setUsage(THREE.DynamicDrawUsage);
  ng.setAttribute('iPos', aPos); ng.setAttribute('iL', aL); ng.instanceCount = 0;
  const nm = new THREE.ShaderMaterial({ vertexShader: NVS, fragmentShader: NFS, uniforms: { ...uni, uPulse: { value: 0 } } });
  const neurons = new THREE.Mesh(ng, nm); neurons.frustumCulled = false; group.add(neurons);

  const maxE = maxNeurons * NET.K;
  const eg = new THREE.BufferGeometry();
  const ePos = new Float32Array(maxE * 6), eL = new Float32Array(maxE * 2);
  const epa = new THREE.BufferAttribute(ePos, 3), ela = new THREE.BufferAttribute(eL, 1);
  // positions are interleaved (2 verts per edge): the typed array holds xyz xyz
  epa.setUsage(THREE.DynamicDrawUsage); ela.setUsage(THREE.DynamicDrawUsage);
  eg.setAttribute('position', epa); eg.setAttribute('aL', ela); eg.setDrawRange(0, 0);
  const em = new THREE.ShaderMaterial({ vertexShader: EVS, fragmentShader: EFS, uniforms: uni, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  const edges = new THREE.LineSegments(eg, em); edges.frustumCulled = false; group.add(edges);

  const info = { cols: 0, neurons: 0, edges: 0, counts: [] };
  const colX = (l, cols) => NET.x0 + (cols <= 1 ? 0 : l * (NET.x1 - NET.x0) / (cols - 1));
  const gridPos = (i, n, out, o) => { const g = Math.ceil(Math.sqrt(n)), r = Math.ceil(n / g); const gy = Math.floor(i / g), gz = i % g; out[o] = (gy - (r - 1) / 2) * NET.spacing; out[o + 1] = (gz - (g - 1) / 2) * NET.spacing; };
  const tmp = [0, 0];

  function rebuild({ width, depth, inDim = 3, outDim = 4 }) {
    const cols = depth + 2, counts = [inDim, ...Array(depth).fill(hiddenCount(width)), outDim];
    let nI = 0, nE = 0; const start = [];
    for (let l = 0; l < cols; l++) {
      start.push(nI); const n = counts[l], x = colX(l, cols);
      for (let i = 0; i < n; i++) {
        gridPos(i, n, tmp, 0);
        iPos[nI * 3] = x; iPos[nI * 3 + 1] = NET.y + tmp[0]; iPos[nI * 3 + 2] = tmp[1]; iL[nI] = l / (cols - 1); nI++;
      }
    }
    for (let l = 0; l < cols - 1; l++) {
      const n = counts[l], m = counts[l + 1], k = Math.min(NET.K, m);
      for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) {
        const t = (i * 7 + j * Math.max(1, Math.floor(m / k)) + (i % 3)) % m, a = (start[l] + i) * 3, b = (start[l + 1] + t) * 3, o = nE * 6;
        ePos[o] = iPos[a]; ePos[o + 1] = iPos[a + 1]; ePos[o + 2] = iPos[a + 2]; ePos[o + 3] = iPos[b]; ePos[o + 4] = iPos[b + 1]; ePos[o + 5] = iPos[b + 2];
        eL[nE * 2] = eL[nE * 2 + 1] = (l + 0.5) / (cols - 1); nE++;
      }
    }
    aPos.needsUpdate = aL.needsUpdate = epa.needsUpdate = ela.needsUpdate = true;
    ng.instanceCount = nI; eg.setDrawRange(0, nE * 2);
    info.cols = cols; info.neurons = nI; info.edges = nE; info.counts = counts;
  }
  return {
    group, info, rebuild,
    setWaves(f, b, o, adam) { uni.uF.value = f; uni.uB.value = b; uni.uO.value = o; uni.uAdam.value = adam ? 1 : 0; },
    colX: (l) => colX(l, info.cols),
    dispose() { base.dispose(); ng.dispose(); nm.dispose(); eg.dispose(); em.dispose(); },
  };
}
