// physics_ui.js - small DOM helpers for the physics scene: playback control bar and the equation panel.
export function makeControls(id, ariaLabel, accent = '#e0408a') {
  const ui = document.createElement('div'); ui.id = id; ui.className = 'card'; ui.setAttribute('role', 'group'); ui.setAttribute('aria-label', ariaLabel);
  Object.assign(ui.style, { display: 'none', top: '86px', left: '12px', padding: '6px 8px', gap: '6px', alignItems: 'center', zIndex: 5, flexWrap: 'wrap', maxWidth: 'min(420px, calc(100vw - 24px))' });
  const mk = (txt, label) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = txt; b.setAttribute('aria-label', label); Object.assign(b.style, { background: '#121a2b', color: '#e8edf5', border: '1px solid #26314a', borderRadius: '8px', padding: '5px 10px', cursor: 'pointer', font: 'inherit' }); return b; };
  const replay = mk('↻ Replay', 'Replay animation'), play = mk('❚❚ Pause', 'Pause or play animation');
  const scrub = document.createElement('input'); scrub.type = 'range'; scrub.min = 0; scrub.max = 1000; scrub.value = 0; scrub.setAttribute('aria-label', 'Animation progress');
  Object.assign(scrub.style, { width: '110px', accentColor: accent });
  const note = document.createElement('span'); Object.assign(note.style, { color: '#8793a6', fontSize: '12px' });
  if (innerWidth <= 640) note.style.display = 'none';
  ui.append(replay, play, scrub, note); document.body.appendChild(ui);
  const place = () => { const ht = document.getElementById('hud-top'); if (ht) ui.style.top = Math.round(ht.getBoundingClientRect().bottom + 6) + 'px'; };
  let was = false; addEventListener('resize', place);
  return { el: ui, replay, play, scrub, note, show(on) { if (on && !was) place(); was = on; ui.style.display = on ? 'flex' : 'none'; }, dispose() { removeEventListener('resize', place); ui.remove(); } };
}

/** Floating card with arbitrary HTML children; returns the element. */
export function makePanel(id, ariaLabel) {
  const p = document.createElement('aside'); p.id = id; p.className = 'card'; p.setAttribute('aria-label', ariaLabel);
  Object.assign(p.style, { display: 'none', zIndex: 4, padding: '8px 10px', fontSize: '12px', lineHeight: '1.35', pointerEvents: 'none' });
  document.body.appendChild(p); return p;
}

/** Full-screen SVG overlay with one leader line (pointer-events none). setLine(x1,y1,x2,y2,color) / hide(). */
export function makeLeader(id) {
  const NS = 'http://www.w3.org/2000/svg', svg = document.createElementNS(NS, 'svg'); svg.id = id;
  Object.assign(svg.style, { position: 'fixed', inset: '0', width: '100%', height: '100%', pointerEvents: 'none', zIndex: 3, display: 'none' });
  svg.setAttribute('aria-hidden', 'true');
  const line = document.createElementNS(NS, 'line'), dot = document.createElementNS(NS, 'circle');
  line.setAttribute('stroke-width', '1.6'); line.setAttribute('stroke-dasharray', '5 4'); dot.setAttribute('r', '5'); dot.setAttribute('fill', 'none'); dot.setAttribute('stroke-width', '2');
  svg.append(line, dot); document.body.appendChild(svg);
  return {
    el: svg,
    setLine(x1, y1, x2, y2, color = '#e0408a', opacity = 1) {
      svg.style.display = 'block'; svg.style.opacity = opacity;
      line.setAttribute('x1', x1); line.setAttribute('y1', y1); line.setAttribute('x2', x2); line.setAttribute('y2', y2); line.setAttribute('stroke', color);
      dot.setAttribute('cx', x2); dot.setAttribute('cy', y2); dot.setAttribute('stroke', color);
    },
    hide() { svg.style.display = 'none'; }, dispose() { svg.remove(); },
  };
}
