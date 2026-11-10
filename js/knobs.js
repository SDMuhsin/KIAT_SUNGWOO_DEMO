// knobs.js - knob registry + panel UI. Pure DOM, no THREE.
// define({id,label,type?,min,max,step,value,unit?,group,help,scale?,options?}):
//   type 'slider' (default) | 'toggle' (boolean value) | 'select' (options:[{value,label}] or strings).
//   scale:'log' makes the slider logarithmic (min must be >0); step then snaps the value (e.g. step 8).
// on(id, fn(value,id)) fires on every change (slider drags fire continuously). get(id), set(id,v).
// The panel shows only knobs listed in the current stage's `knobs` (setVisible(ids)).

/** Core knobs shared by model.js/memory/sage. Scenes may define() more at creation time. */
export const CORE_KNOBS = [
  { id: 'batch', label: 'Batch (points/step)', min: 64, max: 262144, step: 64, value: 4096, scale: 'log', group: 'Network', help: 'Collocation points per training step.' },
  { id: 'width', label: 'Layer width', min: 16, max: 1024, step: 8, value: 128, scale: 'log', group: 'Network', help: 'Neurons per hidden layer.' },
  { id: 'depth', label: 'Hidden layers', min: 1, max: 12, step: 1, value: 4, group: 'Network', help: 'Number of hidden layers (linear layers = hidden + 1).' },
  { id: 'precision', label: 'Precision', type: 'select', options: [{ value: 'fp32', label: 'fp32' }, { value: 'bf16', label: 'bf16' }], value: 'fp32', group: 'Hardware', help: 'Storage dtype of weights/activations/gradients (Adam state stays fp32).' },
  { id: 'bandwidth', label: 'Memory bandwidth', min: 100, max: 3000, step: 10, value: 696, unit: 'GB/s', scale: 'log', group: 'Hardware', help: 'Illustrative; default is A40-like.' },
  { id: 'peakTflops', label: 'Peak compute', min: 1, max: 200, step: 1, value: 37, unit: 'TFLOP/s', scale: 'log', group: 'Hardware', help: 'Illustrative; default is A40-like fp32.' },
  { id: 'pdeOrder', label: 'PDE derivative order', min: 1, max: 3, step: 1, value: 2, group: 'Physics', help: 'Highest derivative of the outputs w.r.t. inputs in the residual.' },
  { id: 'nResiduals', label: 'Residual terms', min: 1, max: 8, step: 1, value: 4, group: 'Physics', help: 'Number of PDE residual terms (continuity, momentum, k, epsilon, ...).' },
  { id: 'sageHalf', label: 'Half-width storage (bf16)', type: 'toggle', value: false, group: 'SAGE', help: 'SAGE optimisation: store intermediates at half width.' },
  { id: 'sageSinglePass', label: 'Single-pass bookkeeping', type: 'toggle', value: false, group: 'SAGE', help: 'SAGE optimisation: one pass of bookkeeping.' },
  { id: 'showProjected', label: 'Show projected (bedrock)', type: 'toggle', value: true, group: 'Results', help: 'Bedrock ~40x is a PROJECTION, not a measurement.' },
];

export function createKnobs(panelEl) {
  const defs = new Map(), vals = new Map(), subs = new Map(), els = new Map();
  let visible = new Set();
  const body = panelEl?.querySelector('.knob-body') ?? panelEl;

  const fmt = (d, v) => (d.type === 'toggle' ? (v ? 'on' : 'off') : d.type === 'select' ? String(v) : (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en') : String(+Number(v).toPrecision(4))) + (d.unit ? ' ' + d.unit : ''));
  const snap = (d, v) => { if (d.type === 'slider' || !d.type) { if (d.step) v = Math.round(v / d.step) * d.step; v = Math.min(d.max, Math.max(d.min, v)); } return v; };
  const toPos = (d, v) => (d.scale === 'log' ? Math.log(v / d.min) / Math.log(d.max / d.min) : (v - d.min) / (d.max - d.min));
  const fromPos = (d, t) => snap(d, d.scale === 'log' ? d.min * Math.pow(d.max / d.min, t) : d.min + t * (d.max - d.min));

  function build(d) {
    const row = document.createElement('div'); row.className = 'knob'; row.dataset.knob = d.id; row.dataset.group = d.group || '';
    const lab = document.createElement('label'); lab.htmlFor = 'knob-' + d.id; lab.textContent = d.label;
    const out = document.createElement('output'); out.htmlFor = 'knob-' + d.id;
    let input;
    if (d.type === 'toggle') { input = document.createElement('input'); input.type = 'checkbox'; input.role = 'switch'; input.checked = !!vals.get(d.id); input.onchange = () => set(d.id, input.checked); }
    else if (d.type === 'select') {
      input = document.createElement('select');
      for (const o of d.options) { const op = document.createElement('option'); const ov = typeof o === 'string' ? { value: o, label: o } : o; op.value = ov.value; op.textContent = ov.label; input.appendChild(op); }
      input.value = vals.get(d.id); input.onchange = () => set(d.id, input.value);
    } else {
      input = document.createElement('input'); input.type = 'range'; input.min = 0; input.max = 1000; input.step = 1;
      input.value = Math.round(toPos(d, vals.get(d.id)) * 1000); input.oninput = () => set(d.id, fromPos(d, input.value / 1000), true);
    }
    input.id = 'knob-' + d.id; if (d.help) { input.setAttribute('aria-description', d.help); row.title = d.help; }
    row.append(lab, input, out);
    els.set(d.id, { row, input, out });
    out.textContent = fmt(d, vals.get(d.id));
    return row;
  }
  function render() {
    if (!body) return;
    const groups = new Map();
    for (const [id, d] of defs) { if (!visible.has(id)) continue; const g = d.group || 'Other'; if (!groups.has(g)) groups.set(g, []); groups.get(g).push(id); }
    body.replaceChildren();
    // the stage's headline knobs (SAGE / Results) go first so they are not below the fold of the scrolling panel
    const order = (g) => (g === 'SAGE' || g === 'Results' ? 0 : 1);
    for (const [g, ids] of [...groups].sort((a, b) => order(a[0]) - order(b[0]))) {
      const sec = document.createElement('section'); sec.className = 'knob-group';
      const h = document.createElement('h3'); h.textContent = g; sec.appendChild(h);
      for (const id of ids) sec.appendChild(els.get(id).row);
      body.appendChild(sec);
    }
    panelEl?.classList.toggle('empty', visible.size === 0 || groups.size === 0);
  }

  function define(d) {
    d = { type: 'slider', ...d };
    if (defs.has(d.id)) return; // idempotent: first definition wins, value preserved
    defs.set(d.id, d); vals.set(d.id, d.value); build(d);
  }
  function get(id) { return vals.get(id); }
  function set(id, v, fromUI = false) {
    const d = defs.get(id); if (!d) throw new Error('unknown knob ' + id);
    if (d.type === 'toggle') v = !!v; else if (d.type !== 'select') v = snap(d, +v);
    if (vals.get(id) === v) return; vals.set(id, v);
    const e = els.get(id); e.out.textContent = fmt(d, v);
    if (!fromUI) { if (d.type === 'toggle') e.input.checked = v; else if (d.type === 'select') e.input.value = v; else e.input.value = Math.round(toPos(d, v) * 1000); }
    for (const fn of subs.get(id) ?? []) fn(v, id);
    for (const fn of subs.get('*') ?? []) fn(v, id);
  }
  function on(id, fn) { if (!subs.has(id)) subs.set(id, new Set()); subs.get(id).add(fn); return () => subs.get(id).delete(fn); }
  function setVisible(ids) { visible = new Set(ids); render(); }
  CORE_KNOBS.forEach(define);
  return { define, get, set, on, setVisible, has: (id) => defs.has(id), ids: () => [...defs.keys()], visibleIds: () => [...visible] };
}
