// physics_cam.js - copy of memory_cam.js frameBox plus an optional `reserve` {top,left,right,bottom} (px) for extra DOM panels.
// (below the title card, left of the knob panel, above the bottom card). Pure function of THREE + camera + DOM rects.
export function frameBox(THREE, camera, box, { az, el, reserve = {} } = {}) {
  const W = innerWidth, H = innerHeight, mobile = W <= 640;
  if (az === undefined) az = mobile ? -50 : -22;
  if (el === undefined) el = mobile ? 28 : 18;
  const [x0, x1, y0, y1, z0, z1] = box;
  const rect = (id) => document.getElementById(id)?.getBoundingClientRect();
  const top = (rect("hud-top")?.bottom ?? 80) + 8 + (reserve.top ?? 0), cardTop = rect('hud-bottom')?.top ?? H - 170;
  let left = 8 + (reserve.left ?? 0), right = W - 8, bottom = cardTop - 8;
  if (mobile) bottom = Math.min(bottom, H - 250 - (reserve.bottom ?? 0)); else { right = W - 330 - (reserve.right ?? 0); bottom -= (reserve.bottom ?? 0); }
  const a = az * Math.PI / 180, e = el * Math.PI / 180, tv = Math.tan(camera.fov * Math.PI / 360);
  const dir = new THREE.Vector3(Math.sin(a) * Math.cos(e), Math.sin(e), Math.cos(a) * Math.cos(e));
  const rightV = new THREE.Vector3(Math.cos(a), 0, -Math.sin(a)), upV = new THREE.Vector3().crossVectors(dir, rightV).normalize();
  const c = new THREE.Vector3((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), cam = new THREE.PerspectiveCamera(camera.fov, W / H, 0.1, 400), pv = new THREE.Vector3();
  const ux = (left + right) / 2, uy = (top + bottom) / 2, uw = right - left, uh = bottom - top; let d = 30;
  for (let it = 0; it < 7; it++) {
    cam.position.copy(c).addScaledVector(dir, d); cam.lookAt(c); cam.updateMatrixWorld(); cam.updateProjectionMatrix();
    let xa = 1e9, xb = -1e9, ya = 1e9, yb = -1e9;
    for (const X of [x0, x1]) for (const Y of [y0, y1]) for (const Z of [z0, z1]) {
      pv.set(X, Y, Z).project(cam); const sx = (pv.x * 0.5 + 0.5) * W, sy = (0.5 - pv.y * 0.5) * H;
      xa = Math.min(xa, sx); xb = Math.max(xb, sx); ya = Math.min(ya, sy); yb = Math.max(yb, sy);
    }
    const wpp = 2 * d * tv / H, k = Math.max((xb - xa) / uw, (yb - ya) / uh) * 1.02;
    c.addScaledVector(rightV, -(ux - (xa + xb) / 2) * wpp).addScaledVector(upV, (uy - (ya + yb) / 2) * wpp);
    d = Math.min(70, d * k);
  }
  const p = c.clone().addScaledVector(dir, d);
  return { pos: [p.x, p.y, p.z], target: [c.x, c.y, c.z] };
}
