// sage_kit.js - shared helpers of the SAGE scenes: allocation-free instanced boxes, sprite-label helper, injected CSS, world zones.
export const ZONES = { idea: [4, 44, 0], overhead: [4, 66, 0], jax: [4, 88, 0] }; // stacked above the physics zone (y=22): the camera climbs one continuous column of space, nothing is far away

const CSS = `
.sg-panel{position:fixed;z-index:4;font-size:12px;line-height:1.35;color:#e8edf5;overflow:auto}
.sg-panel h2{margin:0 0 4px;font-size:13px;font-weight:650}
.sg-pill{display:inline-block;font-size:11px;letter-spacing:.04em;text-transform:uppercase;padding:1px 8px;border-radius:99px;border:1px solid currentColor;margin-right:6px}
.sg-code{margin:6px 0;padding:6px 8px;background:#0a0e16;border:1px solid #26314a;border-radius:8px;font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre;overflow:hidden;color:#cfe3ff;min-height:8.2em}
.sg-code i{font-style:normal;color:#f2c14e}.sg-code u{text-decoration:none;color:#8793a6}
.sg-chips{display:flex;flex-wrap:wrap;gap:4px;margin:4px 0}
.sg-chip{font-size:11px;padding:1px 7px;border-radius:99px;border:1px solid var(--c);color:var(--c);opacity:.45;transition:opacity .15s,background .15s;white-space:nowrap}
.sg-chip.on{opacity:1;background:color-mix(in srgb,var(--c) 28%,transparent)}
.sg-note{color:#8793a6;font-size:11px}
.sg-note b{color:#f2c14e;font-weight:600}
.sg-chipbtn{font:inherit;font-size:12px;padding:5px 10px;border-radius:99px;border:1px solid #26314a;background:#121a2b;color:#e8edf5;cursor:pointer}
.sg-chipbtn[aria-pressed=true]{border-color:#2fd0e6;background:#0e3a44;color:#e8fbff}
.sg-bar{height:14px;border-radius:4px;overflow:hidden;display:flex;background:#0e1422;border:1px solid #26314a}
.sg-bar>i{display:block;height:100%;transition:width .25s}
.sg-row{margin:8px 0}.sg-row .t{display:flex;justify-content:space-between;gap:8px;font-size:12px}
.sg-row .t b{font-variant-numeric:tabular-nums}
.sg-big{font-size:28px;font-weight:700;line-height:1;color:#2fd0e6;font-variant-numeric:tabular-nums}
.sg-grid{display:grid;gap:10px;grid-template-columns:1fr 1fr}
.sg-card{border:1px solid #26314a;border-radius:10px;padding:8px 10px;background:rgba(10,14,22,.55)}
.sg-foot{font-size:10.5px;line-height:1.35;color:#8793a6;margin-top:8px}.sg-foot p{margin:2px 0}
@media (max-width:640px){.sg-grid{grid-template-columns:1fr}.sg-big{font-size:22px}.sg-code{font-size:10px}}
`;
let styleEl = null, styleUsers = 0;
export function useStyle() { if (!styleEl) { styleEl = document.createElement('style'); styleEl.id = 'sage-style'; styleEl.textContent = CSS; document.head.appendChild(styleEl); } styleUsers++; return () => { if (--styleUsers <= 0 && styleEl) { styleEl.remove(); styleEl = null; styleUsers = 0; } }; }

/** Instanced unit boxes written straight into the matrix/colour arrays (no allocation per frame). set(i,x,y,z,sx,sy,sz,hex); hide(i). */
export function createBoxes(THREE, cap, { opacity = 1, rough = 0.55, emissive = 0.18 } = {}) {
  const geo = new THREE.BoxGeometry(1, 1, 1);
  const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity }); void rough; void emissive; // unlit flat colours: exact palette on the dark background
  const mesh = new THREE.InstancedMesh(geo, mat, cap); mesh.frustumCulled = false;
  const col = new THREE.Color(); mesh.setColorAt(0, col);
  const a = mesh.instanceMatrix.array, ca = mesh.instanceColor.array;
  for (let i = 0; i < cap; i++) { a[i * 16] = a[i * 16 + 5] = a[i * 16 + 10] = 1e-5; a[i * 16 + 15] = 1; } // all hidden until set()
  mat.userData.baseOpacity = opacity;
  const api = {
    mesh, mat, cap, used: 0,
    set(i, x, y, z, sx, sy, sz, hex) {
      const o = i * 16; a[o] = Math.max(sx, 1e-5); a[o + 5] = Math.max(sy, 1e-5); a[o + 10] = Math.max(sz, 1e-5); a[o + 12] = x; a[o + 13] = y; a[o + 14] = z;
      if (hex !== undefined) { col.setHex(hex); ca[i * 3] = col.r; ca[i * 3 + 1] = col.g; ca[i * 3 + 2] = col.b; }
      if (i >= api.used) api.used = i + 1;
    },
    setRGB(i, r, g, b) { ca[i * 3] = r; ca[i * 3 + 1] = g; ca[i * 3 + 2] = b; },
    hide(i) { const o = i * 16; a[o] = a[o + 5] = a[o + 10] = 1e-5; },
    commit() { mesh.count = Math.max(1, api.used); mesh.instanceMatrix.needsUpdate = true; mesh.instanceColor.needsUpdate = true; },
    dispose() { geo.dispose(); mat.dispose(); mesh.dispose(); },
  };
  return api;
}

/** Sprite label owner: add(text,color,size) creates a hidden sprite in `parent`; show(sp,a) sets visibility/opacity (a=0 hides). */
export function createLabels(THREE, util, parent) {
  const all = [];
  return {
    add(text, color = '#e8edf5', size = 0.5, opts = {}) { const s = util.makeLabel(THREE, text, { color, size, ...opts }); s.visible = false; s.material.opacity = 0; s.userData.size = size; parent.add(s); all.push(s); return s; },
    show(sp, a) { const v = a > 0.02; if (sp.visible !== v) sp.visible = v; if (v) sp.material.opacity = a; },
    set(sp, text) { util.setLabel(sp, text); },
    all,
  };
}
export const hexCss = (h) => '#' + h.toString(16).padStart(6, '0');
