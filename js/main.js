// main.js - boot: renderer, camera, ctx, scene modules, HUD, render loop.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as theme from './theme.js';
import * as model from './model.js';
import * as util from './util.js';
import { createKnobs } from './knobs.js';
import { createStageController, STAGES } from './stage.js';
import { installExtras } from './extras.js';
import { createCards } from './cards.js';

const $ = (id) => document.getElementById(id);
const canvas = $('stage');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene(); scene.background = new THREE.Color(theme.BG);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 400); camera.position.set(0, 5, 16);
const controls = new OrbitControls(camera, canvas); controls.enableDamping = true; controls.dampingFactor = 0.08; controls.maxDistance = 60;
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(6, 12, 8); scene.add(sun);

const mq = matchMedia('(prefers-reduced-motion: reduce)');
const state = { reducedMotion: mq.matches, reveal: {}, time: 0 };
mq.addEventListener?.('change', () => { state.reducedMotion = mq.matches; });

// HUD API handed to scenes
const BADGE = { illustrative: ['illustrative', 'illustrative'], 'model estimate': ['model estimate', 'estimate'], projected: ['projected', 'projected'] };
const readouts = new Map();
const hud = {
  /** Replace the provenance badges. Controller sets these from STAGES[].badges; scenes rarely need it. */
  setBadges(list) { const el = $('hud-badges'); el.replaceChildren(...list.map((b) => { const s = document.createElement('span'); s.className = 'badge ' + BADGE[b][1]; s.textContent = BADGE[b][0]; return s; })); },
  /** Show/update a live number: hud.readout('ai', 'Arith. intensity', '12.3 FLOP/B', 'model estimate'). Cleared on stage change. */
  readout(id, label, value, badge) {
    let r = readouts.get(id);
    if (!r) { r = document.createElement('div'); r.className = 'readout'; r.innerHTML = '<span></span><b></b>'; readouts.set(id, r); $('hud-readouts').appendChild(r); }
    r.firstChild.textContent = label; r.lastChild.textContent = value;
    r.querySelector('em')?.remove(); if (badge) { const e = document.createElement('em'); e.textContent = badge; r.lastChild.appendChild(e); }
  },
  clearReadouts() { readouts.clear(); $('hud-readouts').replaceChildren(); },
  toast(msg, ms = 4000) { const t = $('toast'); t.textContent = msg; t.style.display = 'block'; setTimeout(() => { t.style.display = 'none'; }, ms); },
};

const knobs = createKnobs($('knob-panel'));
$('knob-toggle').onclick = () => { const p = $('knob-panel'); const c = p.classList.toggle('collapsed'); $('knob-toggle').setAttribute('aria-expanded', String(!c)); };
if (matchMedia('(max-width:640px)').matches) $('knob-panel').classList.add('collapsed');

/** ctx handed to every scene module. params() = model cfg from current knob values. */
const ctx = { THREE, scene, camera, controls, theme, knobs, model, util, state, hud, params: () => model.paramsFromKnobs(knobs.get) };

function resize() {
  const w = innerWidth, h = innerHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();

const MODULES = ['geometry', 'network', 'memory', 'physics', 'sage'];
const modules = {}; ctx.modules = modules; // cross-module access, e.g. ctx.modules.geometry.api
for (const id of MODULES) {
  try {
    const mod = await import(`./scenes/${id}.js`);
    const m = mod.default(ctx); modules[id] = m; scene.add(m.group);
  } catch (e) { console.error(`scene module "${id}" failed`, e); hud.toast(`Scene "${id}" failed to load (see console)`); }
}

const hudEls = { title: $('hud-title'), caption: $('hud-caption'), bridge: $('hud-bridge'), count: $('hud-count'), bar: $('hud-bar'), prev: $('btn-prev'), next: $('btn-next'), dots: $('hud-dots'), swipe: $('hud-bottom') };

// "Where you came from" thumbnail (bottom card): a small buffer is refreshed from the settled canvas every ~40 idle frames;
// on navigation it is copied to the visible thumbnail, so the previous scene stays on screen as a shrunken anchor. Click = go back.
const thumbBuf = document.createElement('canvas'); thumbBuf.width = 128; thumbBuf.height = 72;
const thumb = $('hud-thumb'), thumbBtn = $('hud-thumb-btn'); let thumbValid = false, thumbIdle = 0;
function grabThumb() {
  try {
    const g = thumbBuf.getContext('2d'), W = canvas.width, H = canvas.height, k = Math.max(128 / W, 72 / H), w = W * k, h = H * k;
    g.drawImage(canvas, (128 - w) / 2, (72 - h) / 2, w, h); thumbValid = true;
  } catch (e) { thumbValid = false; }
}
const hooks = { leave() { if (!thumbValid) return; thumb.getContext('2d').drawImage(thumbBuf, 0, 0); thumbBtn.hidden = false; } };
const cards = createCards();
const controller = createStageController({ ctx, modules, hudEls, cards, hooks });
thumbBtn.onclick = () => { if (controller.from >= 0) controller.goTo(controller.from); };

const extras = installExtras({ controller, STAGES, hud });

// frame-time self-check: CPU ms spent in update() and render() per frame (GPU time is NOT measured; headless runs use software GL)
let last = performance.now(), frames = 0, rec = null;
function loop(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now; frames++;
  const t0 = performance.now();
  controller.update(dt); extras.update(dt);
  const t1 = performance.now();
  renderer.render(scene, camera);
  const t2 = performance.now();
  if (controller.settled) { if (++thumbIdle % 40 === 5) grabThumb(); } else thumbIdle = 0;
  if (rec) { const i = renderer.info; rec.n++; rec.upd += t1 - t0; rec.ren += t2 - t1; rec.max = Math.max(rec.max, t2 - t0); rec.calls += i.render.calls; rec.tris += i.render.triangles; rec.pts += i.render.points; }
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
/** window.__demo.perf(seconds = 3) -> Promise<{stage, frames, cpuMs, updateMs, renderMs, maxMs, calls, triangles, points, geometries, textures, programs}> (averages over the run). */
function perf(seconds = 3) {
  return new Promise((res) => {
    rec = { n: 0, upd: 0, ren: 0, max: 0, calls: 0, tris: 0, pts: 0 };
    setTimeout(() => {
      const r = rec, n = Math.max(1, r.n), i = renderer.info; rec = null;
      const f = (x) => Math.round(x * 100) / 100;
      res({ stage: controller.index + 1, frames: r.n, cpuMs: f((r.upd + r.ren) / n), updateMs: f(r.upd / n), renderMs: f(r.ren / n), maxMs: f(r.max),
        calls: Math.round(r.calls / n), triangles: Math.round(r.tris / n), points: Math.round(r.pts / n), geometries: i.memory.geometries, textures: i.memory.textures, programs: i.programs?.length ?? 0 });
    }, seconds * 1000);
  });
}
/** Test/debug handle: window.__demo */
window.__demo = { ctx, controller, STAGES, modules, renderer, extras, perf, get frames() { return frames; } };
