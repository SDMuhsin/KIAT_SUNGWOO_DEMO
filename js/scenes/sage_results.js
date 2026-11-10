// sage_results.js - stage 15 (sage-results): honest results panel (DOM) over an empty 3D backdrop. All headline numbers come from model.FACTS via
// sage_math.headline()/runRows(); the Amdahl what-if is a MODEL (setup is identical on both backends and is not accelerated).
// Bedrock ~40x is PROJECTED. The retracted 90x is NOT used (only mentioned in a footnote as "corrected to 50x").
import * as SM from './sage_math.js';
import { ZONES, useStyle } from './sage_kit.js';

const Z = ZONES.jax;
const CSS_GREY = '#59647c', CSS_BLUE = '#4a90e2', CSS_CYAN = '#2fd0e6';

export function createResults(ctx, shared) {
  const { util, knobs, hud, model, THREE } = ctx, { clamp } = util;
  const restore = useStyle();
  const H = SM.headline(), rows = SM.runRows(), kStep = model.results().perStep;
  knobs.define({ id: 'setupShare', label: 'Setup share of accelerated run', min: 0, max: 0.95, step: 0.01, value: Math.round(SM.PRESET_SHARE.fine * 100) / 100, group: 'Results', help: 'Share of the accelerated run spent in setup (not accelerated). Drives the Amdahl what-if.' });
  const el = document.createElement('aside'); el.id = 'sage-results'; el.className = 'card sg-panel'; el.setAttribute('aria-label', 'SAGE results'); el.style.display = 'none'; el.style.padding = '10px 12px';
  const runHtml = (r) => {
    const s = SM.setupShareOf(r), seg = s == null ? `<i style="width:100%;background:repeating-linear-gradient(45deg,#2a3550 0 6px,#1b2438 6px 12px)"></i>`
      : `<i style="width:${(s * 100).toFixed(1)}%;background:${CSS_GREY}"></i><i style="width:${((1 - s) * 100).toFixed(1)}%;background:${CSS_BLUE}"></i>`;
    const split = s == null ? 'setup/steps split not reported' : r.id === 'bed' ? `setup 12 min | steps 68 min` : `setup 1.0 min | steps 0.5 min`;
    return `<div class="sg-row" data-row="${r.id}"><div class="t"><span>${r.name}</span><b style="color:${CSS_CYAN}">${r.headline} ${r.projected ? '<span class="sg-pill" style="color:#e0408a">PROJECTED</span>' : ''}</b></div>
      <div class="t sg-note"><span>PyTorch ${r.labelBefore}</span><span>ours ${r.labelAfter}</span></div>
      <div class="sg-bar" title="accelerated run only">${seg}</div><div class="t sg-note"><span>accelerated run: ${split}</span><span>${r.tag}</span></div></div>`;
  };
  el.innerHTML = `
  <div class="sg-grid">
   <div>
    <div class="sg-card" id="sgr-step"><div class="t" style="display:flex;justify-content:space-between;align-items:baseline"><span class="sg-big" id="sgr-big">${H.perStep}</span><span class="sg-note">per training step, same network and schedule</span></div>
      <div class="sg-row"><div class="t"><span>PyTorch, data loading as UofW runs it</span><b>${H.pytorchMs} ms</b></div><div class="sg-bar"><i style="width:100%;background:${CSS_GREY}"></i></div></div>
      <div class="sg-row"><div class="t"><span>Ours (JAX + SAGE)</span><b>${H.oursMs} ms</b></div><div class="sg-bar"><i style="width:${(H.oursMs / H.pytorchMs * 100).toFixed(2)}%;background:${CSS_CYAN}"></i></div></div>
      <div class="sg-note">measured, Oct 2026 deck, one A40</div></div>
    <div class="sg-card" id="sgr-runs" style="margin-top:10px"><div class="sg-note" style="margin-bottom:2px"><b>Whole run</b>: setup is not accelerated, so the gain is smaller than per step</div>${rows.map(runHtml).join('')}</div>
   </div>
   <div>
    <div class="sg-card" id="sgr-amdahl"><div class="t" style="display:flex;justify-content:space-between;align-items:baseline"><b>Why whole run &lt; per step</b><span class="sg-note">model (Amdahl)</span></div>
      <label for="sgr-slider" class="sg-note">Setup share of the accelerated run: <b id="sgr-share" style="color:#f2c14e"></b></label>
      <input id="sgr-slider" type="range" min="0" max="95" step="1" style="width:100%;accent-color:${CSS_CYAN}" aria-describedby="sgr-formula">
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin:4px 0"><button type="button" class="sg-chipbtn" id="sgr-p1">fine-tune ~66%</button><button type="button" class="sg-chipbtn" id="sgr-p2">bedrock ~15%</button></div>
      <div style="display:flex;justify-content:space-between;align-items:baseline"><span class="sg-big" id="sgr-whole"></span><span class="sg-note" id="sgr-formula"></span></div><div class="sg-note" id="sgr-deck"></div>
      <div class="sg-row"><div class="t sg-note"><span>PyTorch run</span><span id="sgr-pt"></span></div><div class="sg-bar" id="sgr-bpt"><i></i><i></i></div></div>
      <div class="sg-row"><div class="t sg-note"><span>Ours</span><span id="sgr-ot"></span></div><div class="sg-bar" id="sgr-bot"><i></i><i></i></div></div>
      <div class="sg-note"><span style="color:${CSS_GREY}">&#9632;</span> setup (same on both) &nbsp; <span style="color:${CSS_BLUE}">&#9632;</span> training steps</div></div>
    <div class="sg-card sg-note" style="margin-top:10px"><b>Credit.</b> Waterloo (UofW) owns the simulation pipeline. USask (Prof. Seok-Bum Ko, Sayed Muhsin) owns the acceleration.</div>
   </div>
  </div>
  <div class="sg-foot" id="sgr-foot">
   <p><b>GPUs.</b> USask A40: 48 GB, 0.70 TB/s. UofW RTX 5080: 16 GB, 0.96 TB/s. Their GPU speeds up PyTorch more than our code, so the speedup there is lower (about 10x vs 16x).</p>
   <p><b>Bedrock.</b> ~51 h is PyTorch time projected from its measured step time; ~1.3 h ours; ~40x is PROJECTED on our A40. On UofW hardware it may be lower; their PyTorch step time is still being measured.</p>
   <p><b>Setup</b> (geometry sampling, data loading, final inference) takes the same time on both backends: 12 min for bedrock, 1 min for fine-tuning.</p>
   <p id="sgr-90"><b>Correction.</b> An earlier 90x (Sep 2026) was corrected to 50x in Oct 2026, once the PyTorch baseline loaded data as UofW runs it.</p>
  </div>`;
  document.body.appendChild(el);
  const $ = (s) => el.querySelector(s), slider = $('#sgr-slider'), S = { vis: 0, key: '', shown: false, frames: 0, setting: false };
  const setShare = (v) => { S.setting = true; knobs.set('setupShare', v); S.setting = false; };
  slider.oninput = () => knobs.set('setupShare', slider.value / 100);
  $('#sgr-p1').onclick = () => setShare(Math.round(SM.PRESET_SHARE.fine * 100) / 100); $('#sgr-p2').onclick = () => setShare(Math.round(SM.PRESET_SHARE.bedrock * 100) / 100);
  const offShare = knobs.on('setupShare', () => { S.key = ''; });
  const offProj = knobs.on('showProjected', () => { S.key = ''; });

  function render() {
    const s = knobs.get('setupShare'), a = SM.amdahlBars(s, kStep), pct = (x) => (x * 100).toFixed(1) + '%';
    if (slider.value != Math.round(s * 100) && document.activeElement !== slider) slider.value = Math.round(s * 100);
    $('#sgr-share').textContent = Math.round(s * 100) + '%';
    $('#sgr-whole').textContent = (a.speedup >= 10 ? a.speedup.toFixed(0) : a.speedup.toFixed(1)) + 'x';
    $('#sgr-formula').textContent = `whole run = s + (1 − s) × ${Math.round(kStep)}; per step is ${H.perStep}`;
    const bp = $('#sgr-bpt').children, bo = $('#sgr-bot').children;
    bp[0].style.cssText = `width:${pct(a.pytorch.setup)};background:${CSS_GREY}`; bp[1].style.cssText = `width:${pct(a.pytorch.steps)};background:${CSS_BLUE}`;
    const ow = a.ours.setup + a.ours.steps; bo[0].style.cssText = `width:${pct(a.ours.setup)};background:${CSS_GREY}`; bo[1].style.cssText = `width:${pct(a.ours.steps)};background:${CSS_CYAN}`;
    $('#sgr-pt').textContent = '100%'; $('#sgr-ot').textContent = `${(ow * 100).toFixed(1)}% of the time (${a.speedup >= 10 ? a.speedup.toFixed(0) : a.speedup.toFixed(1)}x faster)`;
    const proj = !!knobs.get('showProjected'); el.querySelector('[data-row=bed]').style.display = proj ? '' : 'none'; $('#sgr-p2').style.display = proj ? '' : 'none';
    $('#sgr-deck').textContent = Math.abs(s - SM.PRESET_SHARE.fine) < 0.015 ? 'deck: 16x measured (fine-tune, A40); model rounds differently' : Math.abs(s - SM.PRESET_SHARE.bedrock) < 0.015 ? 'deck: ~40x PROJECTED (bedrock)' : '';
    const near = (x) => Math.abs(s - x) < 0.015;
    $('#sgr-p1').setAttribute('aria-pressed', String(near(SM.PRESET_SHARE.fine))); $('#sgr-p2').setAttribute('aria-pressed', String(near(SM.PRESET_SHARE.bedrock)));
  }

  function place(mob) {
    const ht = document.getElementById('hud-top')?.getBoundingClientRect(), hb = document.getElementById('hud-bottom')?.getBoundingClientRect(), st = el.style;
    const top = (ht?.bottom ?? 80) + 8, bottom = (hb?.top ?? innerHeight - 170) - 8;
    st.top = top + 'px'; st.maxHeight = Math.max(160, bottom - top) + 'px'; st.left = mob ? '8px' : '12px'; st.right = mob ? '8px' : '330px'; st.bottom = 'auto'; st.width = 'auto';
  }
  function update(dt, state, cur) {
    const idx = shared.idx.results, r = util.revealOf(state, 'sage.results'), here = state.stageIndex === idx;
    const on = here && r > 0.02; if (on !== S.shown) { S.shown = on; el.style.display = on ? 'block' : 'none'; if (on) { S.key = ''; place(innerWidth <= 640); el.scrollTop = 0; } }
    if (!on) return;
    el.style.opacity = String(clamp(r)); if (S.frames++ % 15 === 0) place(innerWidth <= 640);
    if (S.key !== 'x') { S.key = 'x'; render(); }
    if (cur) {
      hud.readout('ps', 'Per step', `${H.pytorchMs} ms → ${H.oursMs} ms  (${H.perStep})`, 'measured, Oct 2026 deck');
      hud.readout('ft', 'Fine-tune (A40)', `${rows[0].labelBefore} → ${rows[0].labelAfter}  (${rows[0].headline})`, 'measured, Oct 2026 deck');
      hud.readout('ft2', 'Fine-tune (RTX 5080)', `${rows[1].labelBefore} → ${rows[1].labelAfter}  (${rows[1].headline})`, 'measured by UofW');
      if (knobs.get('showProjected')) hud.readout('bd', 'Bedrock (A40)', `${rows[2].labelBefore} → ${rows[2].labelAfter}  (${rows[2].headline})`, 'PROJECTED');
    }
  }
  function frame() { return { pos: [Z[0], Z[1] + 2, Z[2] + 16], target: [Z[0], Z[1], Z[2]] }; }
  const api = { get panelText() { return el.textContent; }, get panel() { return el; }, headline: H, rows, setShare, get shown() { return S.shown; }, wholeSpeedup: () => SM.amdahl(knobs.get('setupShare'), kStep) };
  return { group: new THREE.Group(), update, frame, api, enter() { S.key = ''; },
    dispose() { offShare?.(); offProj?.(); el.remove(); restore(); } };
}
