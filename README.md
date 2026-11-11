# SAGE demo: one PINN training step, and what SAGE removes

A single 3D scene: a large battery pack with a finned heat sink on top, sampled points, the neural network, an illustrative output,
and the autograd graphs (regular training graph, and the much larger physics graph). One training step loops (points, forward, loss,
backward through the graphs, optimizer update). One big **SAGE** button animates the physics graph collapsing into a single thin
"v" lane that feeds one ordinary backward pass. Numbers in the top bar are a **model estimate**; timing and the output colours are **illustrative**.
The only measured figure shown is the Oct deck footnote (about 50x per step, 303 ms to 6.1 ms). The slide deck is in `deck/`.

## Run it on Windows

1. **No install:** double-click `demo\sage-demo.html` (one self-contained file, works offline from `file://`).
2. **With a server:** double-click `start.bat` (needs Python 3 on PATH; it opens http://localhost:8000/demo/index.html).

macOS/Linux: `./start.sh`. Manual: `python -m http.server 8000` in this folder, open http://localhost:8000/demo/index.html.
(Double-clicking `demo\index.html` itself will not work: browsers block ES modules on `file://`. Use `sage-demo.html`.)

## Use it

Drag to orbit, scroll/pinch to zoom. Play/Pause, Replay and the slider scrub the step. The SAGE button (key `S`) toggles with a 3.6 s
animation; the thin slider under it scrubs that transition. The Knobs panel (batch, residual points, width, depth, derivative order, physics
weight, precision, Adam, bandwidth, compute) updates the scene and the readouts live.

## Files

`demo/index.html`, `demo/css/style.css`, `demo/js/` (`main.js` scene + loop, `model.js` cost model, `graph.js` autograd graphs, `net.js`, `geo.js`, `cloud.js`,
`anim.js` step timeline, `knobs.js`), `vendor/three` (Three.js 0.160, offline), `tools/build_single.py` (rebuilds `demo/sage-demo.html`: run it after editing `demo/`).

## Tests

`node tests/model.test.mjs` (model, layout, timeline) and `python tests/smoke.py` (Playwright: http and file:// bundle, SAGE readouts, knobs, scrub,
reduced motion, no GPU resource growth, screenshots into `llmdocs/shots/demo/`). Needs `pip install playwright` and a Chromium.
