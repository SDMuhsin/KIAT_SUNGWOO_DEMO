// physics_pinn.js - stage 8 (pinn): PDE-residual probes, data-vs-physics coverage, loss bars L = L_data + lambda L_physics,
// equation panel, "train" animation (violation decays). All ILLUSTRATIVE. Lives in the OUTPUT duct (where geometry draws the red hotspots).
import * as GM from './geometry_math.js';
import * as PM from './physics_math.js';
import { makeControls, makePanel, makeLeader } from './physics_ui.js';
import { frameBox } from './physics_cam.js';

const NP = 30, NPIN = 44, CELL = 0.5, TRAIN0 = 5.0, TRAIN_DUR = 7.0, T_END = 12.5, MAXSEG = 8;
const OUT_C = [11.5, 0, 0];

function mergedArrow(THREE) { // unit arrow along +x, tail at 0, tip at 1
  const cyl = new THREE.CylinderGeometry(0.18, 0.18, 0.68, 8, 1, false).rotateZ(-Math.PI / 2).translate(0.34, 0, 0).toNonIndexed();
  const cone = new THREE.ConeGeometry(0.5, 0.32, 10, 1, false).rotateZ(-Math.PI / 2).translate(0.84, 0, 0).toNonIndexed();
  const g = new THREE.BufferGeometry();
  for (const k of ['position', 'normal']) { const a = cyl.getAttribute(k).array, b = cone.getAttribute(k).array, o = new Float32Array(a.length + b.length); o.set(a); o.set(b, a.length); g.setAttribute(k, new THREE.BufferAttribute(o, 3)); }
  cyl.dispose(); cone.dispose(); return g;
}

export function createPinn(ctx, shared) {
  const { THREE, util, knobs, theme, hud, camera } = ctx, { clamp, smoothstep: ss } = util, COL = theme.COLORS;
  const group = new THREE.Group(); group.name = 'pinn'; group.visible = false;
  knobs.define({ id: 'lambda', label: 'Physics weight λ', min: 0, max: 10, step: 0.1, value: 1, group: 'Physics', help: 'Loss = L_data + λ·L_physics. At λ = 0 the physics constraint is switched off (data-only).' });
  const cOut = new THREE.Vector3(...OUT_C);

  // ---- probe placement (duct-local -> world)
  const rnd = GM.mulberry32(2026), spots = GM.hotspots(), P = [];
  for (const h of spots) P.push({ x: h.x, y: Math.max(-0.7, Math.min(0.7, h.y)), z: h.z });
  const clear = (x, z) => GM.CELLS.every((c) => (x - c.x) ** 2 + (z - c.z) ** 2 > (GM.R + 0.32) ** 2);
  while (P.length < NP) { const x = (rnd() - 0.5) * 5.2, y = (rnd() - 0.5) * 1.4, z = (rnd() - 0.5) * 2.4; if (clear(x, z)) P.push({ x, y, z }); }
  const W = P.map((p) => 0.35 + 0.65 * GM.hotspotIntensity(spots, p.x, p.z));
  let best = 0; W.forEach((w, i) => { if (w > W[best]) best = i; });
  const pw = P.map((p) => new THREE.Vector3(p.x + cOut.x, p.y, p.z));
  // ---- data pins on boundary
  const pins = [];
  for (let i = 0; i < NPIN; i++) {
    const f = i % 4; let x, y, z;
    if (f === 0) { x = -3; y = (rnd() - 0.5) * 2; z = (rnd() - 0.5) * 3; }            // inlet
    else if (f === 1) { x = (rnd() - 0.5) * 6; y = rnd() < 0.5 ? -1 : 1; z = (rnd() - 0.5) * 3; } // floor / ceiling
    else { x = (rnd() - 0.5) * 6; y = (rnd() - 0.5) * 2; z = f === 2 ? -1.5 : 1.5; }  // side walls
    pins.push(new THREE.Vector3(x + cOut.x, y, z));
  }

  // ---- GPU resources
  const arrowGeo = mergedArrow(THREE);
  const arrowMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95 });
  const arrows = new THREE.InstancedMesh(arrowGeo, arrowMat, NP * 6); arrows.frustumCulled = false; arrows.renderOrder = 6;
  const sph = new THREE.SphereGeometry(1, 12, 9);
  const bubMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.42, depthWrite: false });
  const bubbles = new THREE.InstancedMesh(sph, bubMat, NP); bubbles.frustumCulled = false; bubbles.renderOrder = 7;
  const pinMat = new THREE.MeshBasicMaterial({ color: 0x56b4e9, transparent: true, opacity: 1 });
  const pinMesh = new THREE.InstancedMesh(sph, pinMat, NPIN); pinMesh.frustumCulled = false; pinMesh.renderOrder = 6;
  const boxGeo = new THREE.BoxGeometry(1, 1, 1);
  const volMat = new THREE.MeshBasicMaterial({ color: COL.physics, transparent: true, opacity: 0.07, depthWrite: false, side: THREE.BackSide });
  const vol = new THREE.Mesh(boxGeo, volMat); vol.position.copy(cOut); vol.scale.set(6, 2, 3); vol.renderOrder = 1;
  const cubeEdges = new THREE.EdgesGeometry(boxGeo), edgePos = []; {
    const ea = cubeEdges.getAttribute('position').array;
    for (const p of pw) for (let k = 0; k < ea.length; k += 3) edgePos.push(p.x + ea[k] * CELL, p.y + ea[k + 1] * CELL, p.z + ea[k + 2] * CELL);
  }
  const cellG = new THREE.BufferGeometry(); cellG.setAttribute('position', new THREE.Float32BufferAttribute(edgePos, 3));
  const cellMat = new THREE.LineBasicMaterial({ color: 0xc9d4ea, transparent: true, opacity: 0.55 });
  const cells = new THREE.LineSegments(cellG, cellMat); cells.frustumCulled = false; cells.renderOrder = 5;
  // loss bars: 1 data box + MAXSEG physics segments
  const barMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5, metalness: 0.05, transparent: true, opacity: 0.95, emissive: 0xffffff, emissiveIntensity: 0.18 });
  const bars = new THREE.InstancedMesh(boxGeo, barMat, 1 + MAXSEG); bars.frustumCulled = false; bars.renderOrder = 4;
  bars.setColorAt(0, new THREE.Color(1, 1, 1));
  for (let i = 0; i < 1 + MAXSEG; i++) bars.instanceMatrix.array[i * 16 + 15] = 1;
  const barBase = new THREE.Mesh(boxGeo, new THREE.MeshBasicMaterial({ color: 0x26314a, transparent: true, opacity: 0.8 })); barBase.scale.set(1.6, 0.05, 1.1);
  const BX = OUT_C[0] + 4.9, BY = -1.1, BZ = 0; barBase.position.set(BX, BY - 0.03, BZ);
  // a faint ruler showing the starting height
  group.add(vol, cells, arrows, bubbles, pinMesh, bars, barBase);

  const mkLabel = (txt, color, size = 0.34) => { const s = util.makeLabel(THREE, txt, { color, size }); s.visible = false; group.add(s); return s; };
  const lblData = mkLabel('DATA: a few boundary points', '#56b4e9'), lblPhys = mkLabel('PHYSICS: a probe at every sampled interior point (drawn here in the output duct)', '#ff6fae');
  lblData.position.set(cOut.x, -2.3, 2.1); lblPhys.position.set(cOut.x, 2.55, 2.1);
  const lblSum = mkLabel('L = L_data + λ·L_physics', '#e8edf5', 0.36), lblD = mkLabel('L_data', '#56b4e9', 0.3), lblP = mkLabel('λ·L_physics', '#ff6fae', 0.3);
  lblSum.position.set(BX + 1.0, BY + 2.95, BZ); const lblLam = mkLabel('λ = 1.0', '#f2c14e', 0.3);

  // ---- DOM: controls, equation panel, leader
  const ui = makeControls('phys-controls', 'Training animation', '#e0408a');
  const eq = makePanel('phys-eq', 'PDE residual equations'); const leader = makeLeader('phys-leader');
  const rows = [], head = document.createElement('div'); head.style.cssText = 'color:#8793a6;font-size:11px;margin-bottom:4px';
  head.innerHTML = 'RANS k-ε residual = extra loss at every interior point <span style="color:#ff6fae">●</span> pink bubble = equation violated';
  eq.appendChild(head);
  PM.EQUATIONS.forEach((e) => {
    const r = document.createElement('div'); r.style.cssText = 'display:flex;gap:6px;margin:3px 0;align-items:baseline';
    r.innerHTML = `<i style="flex:none;width:8px;height:8px;border-radius:50%;background:#ff6fae;display:inline-block"></i><span><b style="color:#e8edf5">${e.name}</b> <span style="color:#b9c4d8;font-family:ui-monospace,Menlo,monospace;font-size:11px">${e.tex}</span></span>`;
    eq.appendChild(r); rows.push(r);
  });
  const more = document.createElement('div'); more.style.cssText = 'color:#8793a6;font-size:11px'; eq.appendChild(more);

  // ---- state
  const S = { tc: 0, playing: true, vis: 0, trainP: 0, nRes: -1, lam: -1, mobile: false, framed: false, v: 1 };
  const tmpM = new THREE.Matrix4(), q = new THREE.Quaternion(), pos = new THREE.Vector3(), scl = new THREE.Vector3(), colr = new THREE.Color();
  const qAxis = [new THREE.Quaternion(), new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2), new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -Math.PI / 2)];
  const dirV = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
  const BASE = [0.62, 0.4, 0.4], TH = 0.13;
  const cIn = new THREE.Color(0x8fb4ff), cOk = new THREE.Color(0x3ab795), cBad = new THREE.Color(0xff3d8f), cPin = new THREE.Color(0x56b4e9);
  for (let i = 0; i < NP * 6; i++) arrows.setColorAt(i, cIn); for (let i = 0; i < NP; i++) bubbles.setColorAt(i, cBad); for (let i = 0; i < NPIN; i++) pinMesh.setColorAt(i, cPin);
  const barCols = [new THREE.Color(0x56b4e9)]; for (let i = 0; i < MAXSEG; i++) barCols.push(new THREE.Color(i % 2 ? 0xff5fa5 : 0xc2307a));
  barCols.forEach((c, i) => bars.setColorAt(i, c));
  function syncUI() {
    ui.play.textContent = S.playing ? '❚❚ Pause' : '▶ Play'; if (document.activeElement !== ui.scrub) ui.scrub.value = Math.round(S.tc / T_END * 1000);
    ui.note.textContent = innerWidth <= 640 ? '' : S.tc < 1.6 ? 'data only' : S.tc < TRAIN0 ? '+ physics' : S.tc < T_END - 0.01 ? 'training' : 'trained';
  }
  ui.replay.onclick = () => { S.tc = 0; S.playing = true; syncUI(); };
  ui.play.onclick = () => { if (S.tc >= T_END) S.tc = 0; S.playing = !S.playing; syncUI(); };
  ui.scrub.oninput = () => { S.playing = false; S.tc = ui.scrub.value / 1000 * T_END; syncUI(); };
  const uiTimer = setInterval(() => { if (ui.el.style.display !== 'none') syncUI(); }, 200);

  function layoutPanel(show) {
    const mobile = innerWidth <= 640; S.mobile = mobile;
    eq.style.display = show ? 'block' : 'none';
    if (!show) { leader.hide(); return; }
    const kp = document.getElementById('knob-panel'), kb = kp && !kp.classList.contains('empty') ? kp.getBoundingClientRect() : null;
    if (mobile) {
      const ht = document.getElementById('hud-top').getBoundingClientRect(), cb = ui.el.getBoundingClientRect();
      Object.assign(eq.style, { top: (Math.max(ht.bottom, cb.height ? cb.bottom : 0) + 6) + 'px', left: '12px', right: '12px', width: 'auto', maxHeight: '23vh', overflow: 'hidden', fontSize: '10.5px' });
    } else {
      const top = (kb ? kb.bottom : 12) + 10, bt = document.getElementById('hud-bottom').getBoundingClientRect().top;
      Object.assign(eq.style, { top: top + 'px', right: '12px', left: 'auto', width: '300px', maxHeight: Math.max(120, bt - top - 10) + 'px', overflow: 'auto', fontSize: '12px' });
    }
  }
  function updateRows() {
    const n = knobs.get('nResiduals'), mobile = innerWidth <= 640;
    rows.forEach((r, i) => { r.style.opacity = i < n ? 1 : 0.28; r.style.display = mobile && i >= Math.min(n, 3) ? 'none' : 'flex'; });
    more.textContent = mobile ? (n > 3 ? `+ ${n - 3} more residual terms (${n} active)` : '') : `${n} residual term${n > 1 ? 's' : ''} active (knob)`;
    head.style.display = mobile ? 'none' : 'block';
  }
  knobs.on('nResiduals', () => { S.nRes = -1; updateRows(); });

  // ---- per-frame
  function update(dt, state, cur) { // cur: this stage is current
    const idx = shared.idx.pinn, r = util.revealOf(state, 'pinn.constraints');
    const after = state.stageIndex > idx;
    S.vis += ((after ? 0 : 1) - S.vis) * (state.reducedMotion ? 1 : 1 - Math.exp(-dt * 4));
    if (Math.abs(S.vis - (after ? 0 : 1)) < 0.002) S.vis = after ? 0 : 1;
    const G = r * S.vis; group.visible = G > 0.001;
    const lam = knobs.get('lambda'), nRes = knobs.get('nResiduals');
    if (cur && S.playing && !state.reducedMotion) { S.tc = Math.min(T_END, S.tc + dt); if (S.tc >= T_END) { S.playing = false; syncUI(); } }
    if (cur && state.reducedMotion) S.tc = T_END;
    const p = clamp((S.tc - TRAIN0) / TRAIN_DUR), lc = PM.lossCurves(p, lam);
    S.trainP = p; S.v = lc.violation;
    const geo = shared.geo(); if (geo) geo.setPhysicsViolation(clamp(lc.violation * (lam <= 0 ? 0.95 : 1)));
    ui.show(cur && r > 0.5); layoutPanel(cur && r > 0.5 && G > 0.5);
    if (!group.visible) return;
    // fades
    cellMat.opacity = 0.55 * G; arrowMat.opacity = 0.95 * G; bubMat.opacity = 0.42 * G; volMat.opacity = 0.07 * G * ss(1.4, 3.5, S.tc) * (after ? 0 : 1); barMat.opacity = 0.95 * G;
    barBase.material.opacity = 0.8 * G; pinMat.opacity = G;
    // pins
    for (let i = 0; i < NPIN; i++) { const g = ss(0.2 + i * 0.025, 0.6 + i * 0.025, S.tc) * G; pos.copy(pins[i]); scl.setScalar(0.075 * g + 1e-5); pinMesh.setMatrixAt(i, tmpM.compose(pos, q.identity(), scl)); }
    pinMesh.instanceMatrix.needsUpdate = true;
    // probes
    for (let i = 0; i < NP; i++) {
      const g = ss(1.5 + i * 0.07, 2.0 + i * 0.07, S.tc) * G, res = clamp(W[i] * lc.violation * (lam <= 0 ? 1 : 1));
      const c = pw[i];
      for (let a = 0; a < 3; a++) {
        const d = dirV[a], base = BASE[a] * g, lenOut = base * (1 + 1.25 * res);
        pos.copy(c).addScaledVector(d, -CELL / 2 - base); scl.set(Math.max(base, 1e-5), TH * g + 1e-5, TH * g + 1e-5); arrows.setMatrixAt(i * 6 + a * 2, tmpM.compose(pos, qAxis[a], scl));
        pos.copy(c).addScaledVector(d, CELL / 2); scl.set(Math.max(lenOut, 1e-5), TH * g + 1e-5, TH * g + 1e-5); arrows.setMatrixAt(i * 6 + a * 2 + 1, tmpM.compose(pos, qAxis[a], scl));
        colr.copy(cOk).lerp(cBad, clamp(res * 1.5)); arrows.setColorAt(i * 6 + a * 2 + 1, colr);
      }
      colr.copy(cOk).lerp(cBad, clamp(res * 1.5)); bubbles.setColorAt(i, colr);
      pos.copy(c); scl.setScalar((0.04 + 0.34 * Math.pow(res, 0.8)) * g + 1e-5); bubbles.setMatrixAt(i, tmpM.compose(pos, q.identity(), scl));
    }
    arrows.instanceMatrix.needsUpdate = true; arrows.instanceColor.needsUpdate = true; bubbles.instanceMatrix.needsUpdate = true; bubbles.instanceColor.needsUpdate = true;
    // bars
    const bh = PM.barHeights(p, lam, nRes), bw = 0.7, gb = ss(0.6, 1.8, S.tc) * G;
    pos.set(BX, BY + bh.data * 0.5, BZ); scl.set(bw * gb + 1e-5, Math.max(bh.data * gb, 1e-4), bw * gb + 1e-5); bars.setMatrixAt(0, tmpM.compose(pos, q.identity(), scl));
    const segH = lam * 0.25 * lc.phys * (2.2 / (1 + lam * nRes / 4)); let y = BY + bh.data;
    for (let k = 0; k < MAXSEG; k++) {
      const on = k < nRes && lam > 0.005, h = on ? Math.max(segH * ss(1.6 + k * 0.08, 2.4 + k * 0.08, S.tc), 0) : 0;
      pos.set(BX, y + h * 0.5, BZ); scl.set(bw * 0.94 * gb + 1e-5, Math.max(h * gb - 0.012, 1e-5), bw * 0.94 * gb + 1e-5); bars.setMatrixAt(1 + k, tmpM.compose(pos, q.identity(), scl)); y += h;
    }
    bars.instanceMatrix.needsUpdate = true; bars.instanceColor.needsUpdate = true;
    const show = (l, on) => { l.visible = on; l.material.opacity = on ? G : 0; };
    show(lblData, ss(0.3, 1.0, S.tc) > 0.05 && cur); show(lblPhys, ss(1.5, 2.2, S.tc) > 0.05 && cur); show(lblSum, gb > 0.05); show(lblD, gb > 0.05); show(lblP, gb > 0.05 && lam > 0.005 && y - BY - bh.data > 0.1);
    lblLam.visible = gb > 0.05; lblLam.material.opacity = G; util.setLabel(lblLam, 'λ = ' + lam.toFixed(1));
    lblD.position.set(BX + 1.25, BY + 0.15, BZ); lblP.position.set(BX + 1.35, Math.max(BY + bh.data + (y - BY - bh.data) * 0.5, BY + 0.6), BZ); lblLam.position.set(BX - 1.2, BY + 0.5, BZ);
    if (cur) {
      hud.readout('lam', 'Loss = data + λ·physics', `${bh.data.toFixed(2)} + ${lam.toFixed(1)}×${(lc.phys).toFixed(2)}`, 'illustrative');
      hud.readout('cov', 'Constraint coverage', `${NPIN} boundary pts vs ${NP}+ interior pts`, 'illustrative');
      hud.readout('viol', 'PDE violation', `${Math.round(lc.violation * 100)}%` + (lam <= 0 ? ' (physics off)' : ''), 'illustrative');
      // leader line to the worst probe
      if (!S.mobile || true) {
        const v = pw[best].clone().project(camera), x = (v.x * 0.5 + 0.5) * innerWidth, yy = (0.5 - v.y * 0.5) * innerHeight, er = eq.getBoundingClientRect();
        if (v.z < 1 && er.width > 0 && S.tc > 2.2) { const mob = innerWidth <= 640; leader.setLine(mob ? er.left + er.width / 2 : er.left, mob ? er.bottom : er.top + 24, x, yy, '#ff6fae', Math.min(1, G * 1.2)); } else leader.hide();
      }
    } else leader.hide();
  }
  function enter() { S.tc = ctx.state.reducedMotion ? T_END : 0; S.playing = true; updateRows(); syncUI(); }
  function frame() { // camera box: output duct + bars + part of the net
    const mob = innerWidth <= 640;
    return frameBox(THREE, camera, mob ? [8.2, 17.6, -2.7, 3.6, -2.2, 2.2] : [8.2, 19.2, -2.7, 3.5, -2.2, 2.2], { az: mob ? -35 : -24, el: mob ? 24 : 16, reserve: mob ? { top: Math.round(innerHeight * 0.23) + 48 } : { top: 40 } });
  }
  updateRows();
  return { group, update, enter, frame, api: { get trainProgress() { return S.trainP; }, get violation() { return S.v; }, get clock() { return S.tc; }, setClock(t) { S.tc = clamp(t, 0, T_END); S.playing = false; syncUI(); }, get probeCount() { return NP; }, get pinCount() { return NPIN; }, T_END, probes: pw, bestProbe: best },
    dispose() { clearInterval(uiTimer); ui.dispose(); eq.remove(); leader.dispose(); arrowGeo.dispose(); sph.dispose(); boxGeo.dispose(); cubeEdges.dispose(); cellG.dispose(); [arrowMat, bubMat, pinMat, volMat, cellMat, barMat, barBase.material].forEach((m) => m.dispose()); arrows.dispose(); bubbles.dispose(); pinMesh.dispose(); bars.dispose(); util.disposeObject(group); } };
}
