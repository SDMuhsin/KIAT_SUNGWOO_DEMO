// util.js - shared helpers: easing, sprite labels, disposal.
export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
export const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/**
 * makeLabel(THREE, text, {color='#e8edf5', size=0.5, bg='rgba(11,15,23,0.72)', font}) -> THREE.Sprite (always faces camera).
 * size = world height. Update with setLabel(sprite, newText).
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
