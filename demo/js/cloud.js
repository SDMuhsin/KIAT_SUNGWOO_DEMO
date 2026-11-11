// cloud.js - sampled input points on/around the geometry. Two Points objects (data = pale, residual/physics = magenta) drawn from
// fixed seeded pools; the knob only changes drawRange (no allocation). About a third of the points "travel": during the first phase
// of the step they fly along an arc to the network input layer (all in the vertex shader, driven by one uniform).
import { COLORS } from './theme.js';
import { samplePools } from './geo.js';

export const POOL = 2560;
/** knob value -> number of points drawn (sqrt-ish so 64..262144 maps to 40..2560) */
export const drawCount = (n) => Math.max(8, Math.min(POOL, Math.round(40 * Math.sqrt(n / 64))));

const VS = `
attribute float aSeed; attribute vec3 aDst;
uniform float uFly, uScale, uSize, uPx, uAmp;
varying float vA; varying float vFly;
void main(){
  float travels = step(aSeed, 0.34);
  float t = clamp((uFly - aSeed*0.55) / 0.45, 0.0, 1.0);
  t = travels * t * t * (3.0 - 2.0 * t);
  vec3 src = position;
  vec3 p = mix(src, aDst, t);
  p.y += sin(t * 3.14159) * (1.6 + 2.0 * aSeed);
  p.z *= 1.0 - 0.6 * t;
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = max(1.5, uSize * uScale / -mv.z * (1.0 + 0.8 * sin(t * 3.14159)));
  vFly = t;
  vA = uAmp * (t > 0.995 ? 0.0 : 1.0) * (0.55 + 0.45 * travels);
}`;
const FS = `
precision mediump float;
uniform vec3 uColor; varying float vA; varying float vFly;
void main(){
  vec2 q = gl_PointCoord - 0.5; float r = dot(q, q);
  if (r > 0.25 || vA < 0.01) discard;
  gl_FragColor = vec4(uColor * (1.15 - r * 1.6), vA * (0.35 + 0.65 * smoothstep(0.25, 0.05, r)));
  #include <colorspace_fragment>
}`;

export function createCloud(THREE) {
  const target = { x: 0, y: 0, z: 0 };
  const pools = samplePools(POOL);
  const group = new THREE.Group();
  const shared = { uFly: { value: 0 }, uScale: { value: 800 }, uPx: { value: 1 } };
  const make = (positions, color, size, seedOff) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const seed = new Float32Array(POOL), dst = new Float32Array(POOL * 3);
    for (let i = 0; i < POOL; i++) seed[i] = (((i + seedOff) * 0.6180339887) % 1);
    g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1)); g.setAttribute('aDst', new THREE.BufferAttribute(dst, 3));
    const m = new THREE.ShaderMaterial({ vertexShader: VS, fragmentShader: FS, transparent: true, depthWrite: false, depthTest: false,
      uniforms: { ...shared, uColor: { value: new THREE.Color(color) }, uSize: { value: size }, uAmp: { value: 1 } } });
    const pts = new THREE.Points(g, m); pts.frustumCulled = false; pts.renderOrder = 5; group.add(pts);
    return pts;
  };
  const data = make(pools.data, COLORS.points, 0.12, 0);
  const res = make(pools.res, COLORS.physics, 0.12, 977);
  return {
    group, data, res,
    setCounts(nd, nr) { data.geometry.setDrawRange(0, nd); res.geometry.setDrawRange(0, nr); },
    setLambda(l) { res.material.uniforms.uAmp.value = Math.min(1, 0.45 + 0.3 * Math.log2(Math.max(l, 0.1) * 4) / 2); },
    /** network input position in this group's local space; the points spread a little around it */
    setTarget(t) {
      for (const pts of [data, res]) {
        const seed = pts.geometry.getAttribute('aSeed').array, a = pts.geometry.getAttribute('aDst');
        for (let i = 0; i < POOL; i++) { a.array[i * 3] = t.x + (seed[i] - 0.5) * 0.2; a.array[i * 3 + 1] = t.y + (((i * 0.37) % 1) - 0.5) * 0.6; a.array[i * 3 + 2] = t.z + (((i * 0.71) % 1) - 0.5) * 0.6; }
        a.needsUpdate = true;
      }
    },
    setFly(f) { shared.uFly.value = f; },
    setViewport(h, fov, pr) { shared.uScale.value = h / (2 * Math.tan(fov / 2)); shared.uPx.value = pr; },
    dispose() { data.geometry.dispose(); res.geometry.dispose(); data.material.dispose(); res.material.dispose(); },
  };
}
