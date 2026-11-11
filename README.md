# SAGE interactive 3D demo

A 20-slide (15 3D stages + 5 short 2D explainer cards), progressive-disclosure 3D walkthrough: a battery-pack cooling duct is sampled into points, fed through a neural
network, the memory and compute cost of the backward pass is shown, physics (PINN) constraints are added, and finally the SAGE
IP removes the overhead. Everything runs locally in the browser (Three.js 0.160.0 is vendored, no network needed).

## Quick start (Windows)

1. **Easiest, no install:** double-click `dist\sage-demo.html`. It is one self-contained file and works offline from `file://`.
2. **With a local server:** double-click `start.bat`. It tries `py`, then `python`, then `npx serve`, picks a free port
   (8000 first), and opens your default browser. Needs Python 3 (https://www.python.org/downloads/, tick "Add python.exe to PATH").

## Three ways to run

| Way | Command | Notes |
|---|---|---|
| Single file | open `dist/sage-demo.html` | No server, no internet. Rebuild after editing source: `python tools/build_single.py` |
| Launcher | `start.bat` (Windows) or `./start.sh` (macOS/Linux) | Uses `tools/serve.py`, a stdlib wrapper over `http.server` that forces correct JS MIME types (some Windows registries serve `.js` as text/plain, which breaks ES modules) |
| Manual | `python -m http.server 8000` in this folder, then open http://localhost:8000/ | Or `npx --yes serve .` |

Why does double-clicking `index.html` not work? Browsers block ES-module imports from `file://` (CORS). Use `dist/sage-demo.html`
(all modules are bundled into one classic script) or one of the server options.

## Controls

| Input | Action |
|---|---|
| Right / Space / PgDn, Left / Shift+Space / PgUp | next / previous stage (reverses cleanly) |
| Home / End | first / last stage |
| `?` or `H` | help overlay (also the `?` button, top-left card) |
| `T` | Auto-tour on/off (next stage every ~12 s; any interaction pauses it, it resumes after 20 s idle) |
| Mouse drag / right-drag / wheel | orbit / pan / zoom. Touch: 1 finger orbit, 2 fingers pan+zoom; swipe on the bottom card changes stage |
| Right-hand panel | sliders and toggles for the current stage (collapsible "Controls" on phones) |
| `#stage=pinn` in the URL | deep link; ids: geometry, sampling, network, forward, memory-fwd, backward, roofline-primer, bounds, pinn-primer, pinn, derivs, graph-primer, phys-grad, autograd-cost, split-primer, sage-idea, bars-primer, sage-overhead, sage-jax, sage-results |

## Stages

1 geometry, 2 sampling points, 3 the network, 4 forward pass, 5 memory: weights + activations, 6 backward pass (gradients, Adam state),
**7 card: bytes vs FLOPs**, 8 compute- vs memory-bound (roofline, drawn just under the memory blocks), **9 card: the PDE residual**, 10 PINN physics constraints,
11 derivatives w.r.t. inputs, **12 card: why the gradient graph grows**, 13 weight gradient vs physics gradient, 14 autograd cost of physics,
**15 card: split the gradient (v and J)**, 16 SAGE: derive v once, **17 card: before/after bars**, 18 SAGE: overhead removed, 19 SAGE: one compiled program, 20 results.
(Bold = 2D card: one idea, SVG over the dimmed 3D scene; the previous 3D stage stays in place behind it. Square dots in the progress bar.)

### Continuity between slides

- One continuous world: the physics gradient graphs and the SAGE stages are stacked straight above the network (y = 22, 44, 66, 88), the roofline sits just under the memory blocks, so the camera always flies through one connected space (longer moves bow outward and take longer).
- On a long camera move the old scene stays visible until the camera has mostly left it and the new one fades in as the camera arrives.
- The bottom card shows a thumbnail of the view you left (click to go back) and a yellow bridge line: "Last: X. Now: Y, because ...".
- Reverse navigation plays the same in reverse ("Back from Y to X").

## What is measured, projected, estimated, illustrative

Every stage carries badges in the title card.

- **Measured (from the Sep/Oct 2026 decks):** per-step 303 ms (PyTorch, UofW data loading) vs 6.1 ms = 50x; fine-tune whole run
  25.4 min to 1.6 min (16x, A40), UofW RTX 5080 about 10x; Sep profiling: about 75% of the step is network+derivative evaluation,
  memory-bound, traffic 126 to 31 units, 12.7 to 7.7 ms/step. Stage 14 labels only those figures "measured".
- **Projected (pink badge):** the bedrock run about 40x (51 h to 1.3 h). Not measured; hide it with the "show projected" toggle.
- **Model estimate (cyan badge):** everything from the analytic cost model `js/model.js` (bytes, FLOPs, arithmetic intensity,
  roofline time, autograd node counts, "x19.6"). Stylised assumptions: 3 us per autograd node, A40-like default bandwidth/peak.
- **Illustrative (gold badge):** geometry, point clouds, network weights/activations (seeded random, 16 neurons drawn per layer),
  the output field (a stylised function, not CFD), all layouts. The retracted 90x figure appears only as a correction.

## Tests

Prerequisites: Python 3, Node.js (for unit tests), `pip install playwright` and `playwright install chromium` (for smoke tests).
The same commands work in Windows `cmd`/PowerShell (use `python` or `py -3`; paths are resolved with `pathlib`).

```
node tests/model.test.mjs                 (also geometry/network/memory/physics/sage .test.mjs)
python tests/run_all.py                   (all node tests + all smoke tests; --skip-smoke for node only)
python tests/package.smoke.py             (vendored/offline run + file:// single-file run, walks all 20 slides)
python tests/smoke.py                     (end-to-end; also geometry/network/memory/physics/sage .smoke.py)
python tests/perf.py                      (frame-time self-check across all 20 slides)
python tests/shots.py OUTDIR --stages 1,4,10   (screenshots at 1280x720 and 390x844)
python tests/transitions.py OUTDIR        (contact sheet of every transition: before / midpoint / settled, both directions)
```

If playwright's browser build does not match, set `CHROMIUM_PATH` to a Chromium/Chrome executable
(Windows example: `set CHROMIUM_PATH=C:\Program Files\Google\Chrome\Application\chrome.exe`). Tests use software GL.
In the browser console, `__demo.perf(3)` returns average CPU ms per frame and `renderer.info` counts over 3 s.

## Troubleshooting

- **Black or blank page / "WebGL not available":** enable hardware acceleration (Chrome/Edge: Settings > System > "Use graphics
  acceleration when available", then restart the browser), update the GPU driver, and check `chrome://gpu`. Remote-desktop sessions
  and some VMs disable WebGL. Try another browser (Edge, Chrome, Firefox).
- **Nothing happens on double-click of `index.html`:** use `dist\sage-demo.html` or `start.bat` (see above).
- **`start.bat` says Python not found:** install Python 3 with "Add to PATH", or just use `dist\sage-demo.html`. The Microsoft
  Store `python` stub does not count; install from python.org.
- **Port busy:** the launcher walks upward from 8000 automatically.
- **Edited the source but `dist` looks old:** run `python tools/build_single.py`.
- **Slow on a laptop:** lower Batch/Width in the Controls panel; stage 4 and 18 are the heaviest. Use `__demo.perf()` to measure.
- **Licenses:** Three.js is MIT (`vendor/three/LICENSE`).

## Layout

`index.html`, `css/`, `js/` (main, stage controller, knobs, cost model, `scenes/`), `vendor/three/` (pinned three.js + OrbitControls),
`tools/` (`serve.py`, `build_single.py`), `dist/sage-demo.html` (generated), `tests/`.
