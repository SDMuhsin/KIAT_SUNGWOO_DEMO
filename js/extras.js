// extras.js - presenter conveniences: help overlay (? or H), auto-tour toggle (T), both wired from main.js.
// Auto-tour advances a stage every TOUR_SECONDS; any user interaction pauses it (it resumes after RESUME_IDLE seconds of no input).
const TOUR_SECONDS = 12, RESUME_IDLE = 20;

export function installExtras({ controller, STAGES, hud }) {
  const $ = (id) => document.getElementById(id);
  const helpBtn = $('btn-help'), tourBtn = $('btn-tour');

  // ---------------- help overlay
  const help = document.createElement('div'); help.id = 'help'; help.setAttribute('role', 'dialog'); help.setAttribute('aria-modal', 'true'); help.setAttribute('aria-label', 'Keyboard and mouse help'); help.hidden = true;
  const rows = [
    ['&rarr; / Space / PgDn', 'next stage'], ['&larr; / Shift+Space / PgUp', 'previous stage'], ['Home / End', 'first / last stage'],
    ['T', 'auto-tour on / off (every ~12 s; pauses on any interaction)'], ['? or H', 'show / hide this help'], ['Esc', 'close help'],
    ['Drag / right-drag / wheel', 'orbit / pan / zoom the 3D scene (touch: 1 finger orbit, 2 fingers pan + zoom)'],
    ['Round dots / square dots', 'round = 3D stage, square = a short 2D explainer card shown over the dimmed scene before a harder stage'],
    ['Thumbnail (bottom card)', 'the view you just left; click it to go back. The yellow line says what came last and why we moved on'],
    ['Sliders (right panel)', 'change the knobs; the scene updates live'], ['Dots / swipe bottom card', 'jump to a stage / swipe left-right'],
    ['URL #stage=pinn', 'deep link to a stage or card (ids: ' + STAGES.map((s) => s.id).join(', ') + ')'],
  ];
  help.innerHTML = '<div class="help-card"><h2>Controls</h2><table>' + rows.map(([k, d]) => `<tr><th>${k}</th><td>${d}</td></tr>`).join('') +
    '</table><p class="help-legend"><span class="badge illustrative">illustrative</span> layout/synthetic visuals &nbsp; <span class="badge estimate">model estimate</span> analytic cost model &nbsp; <span class="badge projected">projected</span> not measured</p>' +
    '<button type="button" id="help-close" class="nav-btn help-x" aria-label="Close help">&times;</button></div>';
  document.body.appendChild(help);
  let lastFocus = null;
  const setHelp = (on) => {
    if (on === !help.hidden) return; help.hidden = !on; helpBtn?.setAttribute('aria-expanded', String(on));
    if (on) { lastFocus = document.activeElement; $('help-close').focus(); } else lastFocus?.focus?.();
  };
  help.addEventListener('click', (e) => { if (e.target === help || e.target.id === 'help-close') setHelp(false); });
  helpBtn && (helpBtn.onclick = () => setHelp(help.hidden));

  // ---------------- auto-tour
  const T = { on: false, t: 0, idle: 0, paused: false };
  const label = () => { if (!tourBtn) return; tourBtn.setAttribute('aria-pressed', String(T.on)); tourBtn.textContent = !T.on ? 'Auto-tour' : T.paused ? 'Auto-tour (paused)' : 'Auto-tour: on'; };
  function setTour(on) { T.on = on; T.t = 0; T.idle = 0; T.paused = false; label(); if (on) hud.toast('Auto-tour: next stage every ~' + TOUR_SECONDS + ' s. Any interaction pauses it.', 2500); }
  tourBtn && (tourBtn.onclick = () => setTour(!T.on));
  const interact = (e) => {
    if (!T.on) return; if (e.target === tourBtn) return;
    T.paused = true; T.idle = 0; T.t = 0; label();
  };
  for (const ev of ['pointerdown', 'wheel', 'touchstart', 'input']) addEventListener(ev, interact, { passive: true, capture: true });
  addEventListener('keydown', (e) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const tag = e.target.tagName, typing = tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA';
    if (e.key === 'Escape') { setHelp(false); return; }
    if (typing) { interact(e); return; }
    if (e.key === '?' || e.key === 'h' || e.key === 'H') { setHelp(help.hidden); e.preventDefault(); }
    else if (e.key === 't' || e.key === 'T') { setTour(!T.on); e.preventDefault(); }
    else interact(e);
  }, true);

  /** Call once per frame with dt seconds. */
  let lastNow = performance.now();
  function update() {
    const now = performance.now(), dt = Math.min(1, (now - lastNow) / 1000); lastNow = now;   // wall-clock (frame dt is capped at 0.1 s on slow machines)
    if (!T.on) return;
    if (T.paused) { T.idle += dt; if (T.idle >= RESUME_IDLE) { T.paused = false; T.t = 0; label(); } return; }
    if (!help.hidden) return;
    T.t += dt;
    if (T.t >= TOUR_SECONDS) {
      T.t = 0;
      if (controller.index >= STAGES.length - 1) { setTour(false); hud.toast('Auto-tour finished.', 2500); } else controller.next();
    }
  }
  label();
  return { update, setHelp, setTour, get tour() { return { ...T }; } };
}
