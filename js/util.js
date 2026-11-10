// util.js - shared helpers for scene modules: easing, reveal gating, instanced fade-in, sprite labels, disposal.
export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
export const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/** Eased reveal (0..1) of a part id, 0 if unknown. Gate every part on this. */
export const revealOf = (state, partId) => state.reveal[partId] ?? 0;

/**
 * gateGroup(obj, r): apply reveal r to any Object3D: visible only if r>0.001, scales 0.8->1, fades every material
 * that has userData.baseOpacity (set by theme.material) or else uses its current opacity as base. Never hard-pops.
 * Cheap enough to call per frame; skips work when r unchanged (stored in obj.userData._r).
 */
export function gateGroup(obj, r) {
  if (obj.userData._r === r) return;
  obj.userData._r = r; obj.visible = r > 0.001;
  if (!obj.visible) return;
  if (obj.userData._s0 === undefined) obj.userData._s0 = obj.scale.clone();
  obj.scale.copy(obj.userData._s0).multiplyScalar(0.8 + 0.2 * r);
  obj.traverse((o) => {
    const ms = o.material ? (Array.isArray(o.material) ? o.material : [o.material]) : [];
    for (const m of ms) { if (m.userData.baseOpacity === undefined) m.userData.baseOpacity = m.opacity; m.transparent = true; m.opacity = m.userData.baseOpacity * r; }
  });
}

/**
 * createFadeInstanced(THREE, geometry, material, count) -> { mesh, set(i,pos,scale=1,color?), commit(), applyReveal(r), dispose }
 * One draw call. Call set() for each instance, then commit(). applyReveal(r) staggers instances: instance i
 * grows in over the window [i/count*0.7, i/count*0.7+0.3] of r, so a cloud "pours in". Only re-uploads when r changes.
 */
export function createFadeInstanced(THREE, geometry, material, count) {
  const mesh = new THREE.InstancedMesh(geometry, material, count);
  mesh.frustumCulled = false; mesh.visible = false;
  const pos = new Float32Array(count * 3), scl = new Float32Array(count).fill(1);
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), v = new THREE.Vector3(), s = new THREE.Vector3(), col = new THREE.Color();
  let last = -1;
  const write = (r) => {
    for (let i = 0; i < count; i++) {
      const k = smoothstep(i / count * 0.7, i / count * 0.7 + 0.3, r) * scl[i];
      v.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]); s.setScalar(Math.max(k, 1e-5));
      mesh.setMatrixAt(i, m4.compose(v, q, s));
    }
    mesh.instanceMatrix.needsUpdate = true;
  };
  return {
    mesh,
    set(i, p, scale = 1, color) { pos[i * 3] = p.x ?? p[0]; pos[i * 3 + 1] = p.y ?? p[1]; pos[i * 3 + 2] = p.z ?? p[2]; scl[i] = scale; if (color !== undefined) mesh.setColorAt(i, col.set(color)); },
    commit() { last = -1; if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true; },
    applyReveal(r) { mesh.visible = r > 0.001; if (r === last || !mesh.visible) return; last = r; write(r); },
    dispose() { mesh.dispose(); },
  };
}

/**
 * makeLabel(THREE, text, {color='#e8edf5', size=0.5, bg='rgba(11,15,23,0.72)', font}) -> THREE.Sprite (always faces camera).
 * size = world height. Update with setLabel(sprite, newText). Opacity is gate-able via gateGroup.
 */
export function makeLabel(THREE, text, opts = {}) {
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ transparent: true, depthWrite: false, depthTest: false }));
  sp.userData.THREE = THREE;
  sp.userData.opts = { color: '#e8edf5', size: 0.5, bg: 'rgba(11,15,23,0.72)', font: '600 40px system-ui, sans-serif', ...opts };
  sp.renderOrder = 10; setLabel(sp, text); return sp;
}
export function setLabel(sp, text) {
  const o = sp.userData.opts; if (sp.userData.text === text) return; sp.userData.text = text;
  const c = document.createElement('canvas'), g = c.getContext('2d'); g.font = o.font;
  const w = Math.ceil(g.measureText(text).width) + 40, h = 72; c.width = w; c.height = h;
  g.font = o.font; g.fillStyle = o.bg; g.beginPath(); g.roundRect(0, 0, w, h, 16); g.fill();
  g.fillStyle = o.color; g.textBaseline = 'middle'; g.fillText(text, 20, h / 2 + 2);
  if (sp.material.map) sp.material.map.dispose();
  sp.material.map = new sp.userData.THREE.CanvasTexture(c); sp.material.map.needsUpdate = true; sp.material.needsUpdate = true;
  sp.scale.set(o.size * w / h, o.size, 1);
}

/** Dispose geometries/materials/textures under obj (call in your module's dispose()). */
export function disposeObject(obj) {
  obj.traverse((o) => {
    o.geometry?.dispose?.();
    const ms = o.material ? (Array.isArray(o.material) ? o.material : [o.material]) : [];
    for (const m of ms) { m.map?.dispose?.(); m.dispose?.(); }
  });
}
