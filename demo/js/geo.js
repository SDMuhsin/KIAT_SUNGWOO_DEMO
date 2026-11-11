// geo.js - the input geometry: a large battery pack (plain box) with a plain finned heat sink on top. Nothing else.
// makePack(THREE, {field}) builds ONE copy; the scene makes two: the grey input and the colour-mapped "illustrative output".
// Also exports PACK (dimensions) and samplePools() which returns the seeded point pools used by cloud.js.
export const PACK = { w: 5.2, h: 0.9, d: 3.0, plateW: 4.8, plateH: 0.1, plateD: 2.8, finN: 13, finL: 4.6, finH: 0.8, finT: 0.07 };
PACK.top = PACK.h / 2;                                  // y of the pack top
PACK.plateTop = PACK.top + PACK.plateH;
PACK.finTop = PACK.plateTop + PACK.finH;
PACK.finZ = (i) => (i / (PACK.finN - 1) - 0.5) * (PACK.plateD - 0.2);

const VS = `
varying vec3 vP; varying vec3 vN;
void main(){
  vec4 lp = vec4(position, 1.0); vec3 n = normal;
  #ifdef USE_INSTANCING
  lp = instanceMatrix * lp; n = mat3(instanceMatrix) * n;
  #endif
  vP = lp.xyz; vN = normalize(mat3(modelMatrix) * n);
  gl_Position = projectionMatrix * modelViewMatrix * lp;
}`;
const FS = `
varying vec3 vP; varying vec3 vN;
uniform vec3 uBase; uniform float uField; uniform float uReveal; uniform float uPhase;
vec3 cmap(float t){ // viridis-like, perceptual, no "temperature" meaning
  vec3 c0=vec3(0.267,0.005,0.329), c1=vec3(0.231,0.318,0.545), c2=vec3(0.128,0.567,0.551), c3=vec3(0.369,0.789,0.383), c4=vec3(0.993,0.906,0.144);
  t=clamp(t,0.,1.)*4.; if(t<1.) return mix(c0,c1,t); if(t<2.) return mix(c1,c2,t-1.); if(t<3.) return mix(c2,c3,t-2.); return mix(c3,c4,t-3.);
}
void main(){
  vec3 n = normalize(vN); if(!gl_FrontFacing) n = -n;
  float d = 0.38 + 0.62 * max(dot(n, normalize(vec3(-0.35, 0.85, 0.5))), 0.0) + 0.12 * abs(n.x);
  vec3 col = uBase;
  if (uField > 0.5) {
    float f = 0.5 + 0.26*sin(0.95*vP.x + 0.7*vP.z + uPhase) + 0.2*cos(1.7*vP.z - 0.6*vP.y) + 0.14*vP.y*0.8;
    float wipe = smoothstep(0.0, 0.6, (uReveal*7.2 - 3.6) - vP.x + 0.3);
    col = mix(uBase*0.8, cmap(f), wipe);
  }
  gl_FragColor = vec4(col * d, 1.0);
  #include <colorspace_fragment>
}`;

function mat(THREE, base, field) {
  return new THREE.ShaderMaterial({ vertexShader: VS, fragmentShader: FS, uniforms: {
    uBase: { value: new THREE.Color(base) }, uField: { value: field ? 1 : 0 }, uReveal: { value: 1 }, uPhase: { value: 0 } } });
}

/** Returns {group, materials, setReveal(r), dispose()}. field=true: colour-mapped output copy. */
export function makePack(THREE, { field = false } = {}) {
  const group = new THREE.Group(), geos = [], mats = [];
  const P = PACK;
  const boxG = new THREE.BoxGeometry(P.w, P.h, P.d, 10, 2, 6), plateG = new THREE.BoxGeometry(P.plateW, P.plateH, P.plateD, 9, 1, 5), finG = new THREE.BoxGeometry(P.finL, P.finH, P.finT, 9, 3, 1);
  geos.push(boxG, plateG, finG);
  const packM = mat(THREE, 0x5b6679, field), sinkM = mat(THREE, 0xc3ccd8, field);
  mats.push(packM, sinkM);
  const pack = new THREE.Mesh(boxG, packM); group.add(pack);
  const plate = new THREE.Mesh(plateG, sinkM); plate.position.y = P.top + P.plateH / 2; group.add(plate);
  const fins = new THREE.InstancedMesh(finG, sinkM, P.finN); fins.frustumCulled = false;
  const m = new THREE.Matrix4();
  for (let i = 0; i < P.finN; i++) { m.makeTranslation(0, P.plateTop + P.finH / 2, P.finZ(i)); fins.setMatrixAt(i, m); }
  fins.instanceMatrix.needsUpdate = true; group.add(fins);
  // crisp outlines so the box reads at a glance
  const edgeG = new THREE.EdgesGeometry(boxG), plateEG = new THREE.EdgesGeometry(plateG);
  const edgeM = new THREE.LineBasicMaterial({ color: 0x9aa7bb, transparent: true, opacity: 0.55 });
  geos.push(edgeG, plateEG); mats.push(edgeM);
  group.add(new THREE.LineSegments(edgeG, edgeM));
  const pe = new THREE.LineSegments(plateEG, edgeM); pe.position.y = plate.position.y; group.add(pe);
  return {
    group, packMat: packM, sinkMat: sinkM,
    setReveal(r) { packM.uniforms.uReveal.value = r; sinkM.uniforms.uReveal.value = r; },
    dispose() { fins.dispose(); geos.forEach((g) => g.dispose()); mats.forEach((x) => x.dispose()); },
  };
}

function rng(seed) { let s = seed >>> 0; return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

/**
 * Seeded point pools in pack-local coordinates. data: boundary points on the surfaces (pack, plate, fin faces);
 * res: collocation points in the volume around the heat sink and a little around the pack. Count is applied with drawRange.
 * Returns {data: Float32Array(n*3), res: Float32Array(n*3)}.
 */
export function samplePools(n = 2560, seed = 7) {
  const r = rng(seed), P = PACK, data = new Float32Array(n * 3), res = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const u = r(); let x, y, z;
    if (u < 0.5) { // fin faces (the dominant boundary)
      const f = Math.floor(r() * P.finN); x = (r() - 0.5) * P.finL; y = P.plateTop + r() * P.finH; z = P.finZ(f) + (r() < 0.5 ? -1 : 1) * P.finT / 2;
    } else if (u < 0.62) { // fin tops
      const f = Math.floor(r() * P.finN); x = (r() - 0.5) * P.finL; y = P.finTop; z = P.finZ(f) + (r() - 0.5) * P.finT;
    } else if (u < 0.78) { // plate top between fins
      x = (r() - 0.5) * P.plateW; y = P.plateTop; z = (r() - 0.5) * P.plateD;
    } else { // pack faces: top, long sides, ends
      const k = r();
      if (k < 0.4) { x = (r() - 0.5) * P.w; y = P.top; z = (r() - 0.5) * P.d; }
      else if (k < 0.7) { x = (r() - 0.5) * P.w; y = (r() - 0.5) * P.h; z = (r() < 0.5 ? -1 : 1) * P.d / 2; }
      else { x = (r() < 0.5 ? -1 : 1) * P.w / 2; y = (r() - 0.5) * P.h; z = (r() - 0.5) * P.d; }
    }
    data.set([x, y, z], i * 3);
    // residual points: volume between/above the fins and a shell around the pack
    const v = r();
    if (v < 0.65) { x = (r() - 0.5) * (P.plateW + 0.8); y = P.plateTop + r() * (P.finH + 0.55); z = (r() - 0.5) * (P.plateD + 0.4); }
    else { x = (r() - 0.5) * (P.w + 1.4); y = -P.h / 2 - 0.35 + r() * (P.h + 0.7); z = (r() - 0.5) * (P.d + 1.0); }
    if (Math.abs(x) < P.w / 2 && Math.abs(y) < P.h / 2 && Math.abs(z) < P.d / 2) y = P.top + 0.05 + r() * 0.4; // never inside the box
    res.set([x, y, z], i * 3);
  }
  return { data, res };
}
