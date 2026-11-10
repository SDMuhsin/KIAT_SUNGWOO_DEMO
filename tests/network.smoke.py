#!/usr/bin/env python3
"""Network scene smoke: python3 tests/network.smoke.py [--shots DIR]. Screenshots stages 3/4 at 1280x720 and 390x844,
zero console errors, knob rebuilds, no growth of renderer.info geometries/textures on repeated width/depth changes."""
import argparse, subprocess, sys, time, socket, os, pathlib, glob
from playwright.sync_api import sync_playwright
ROOT = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument('--shots', default=str(ROOT / 'llmdocs/shots/network')); a = ap.parse_args()
shots = pathlib.Path(a.shots); shots.mkdir(parents=True, exist_ok=True)
s = socket.socket(); s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]; s.close()
srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(port), '--bind', '127.0.0.1'], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(0.8)
errors, fails = [], []
def check(c, m):
    if not c: fails.append(m)
    print('ok  ' if c else 'FAIL', m)
INFO = '({g: __demo.renderer.info.memory.geometries, t: __demo.renderer.info.memory.textures, calls: __demo.renderer.info.render.calls})'
try:
    with sync_playwright() as p:
        la = dict(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
        try: b = p.chromium.launch(**la)
        except Exception:
            c = [os.environ.get('CHROMIUM_PATH')] + sorted(glob.glob(os.path.expanduser('~/.cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell')))
            b = p.chromium.launch(executable_path=next(x for x in c if x), **la)
        for w, h, tag in ((1280, 720, 'desk'), (390, 844, 'mobile')):
            pg = b.new_page(viewport={'width': w, 'height': h})
            pg.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
            pg.on('pageerror', lambda e: errors.append(str(e)))
            pg.goto(f'http://127.0.0.1:{port}/index.html'); pg.wait_for_function('window.__demo && window.__demo.frames > 5', timeout=60000)
            api = '__demo.ctx.modules.network.api'
            def go(stage, wait=4200):
                pg.evaluate(f"__demo.controller.goTo('{stage}')"); pg.wait_for_timeout(wait)
            go('sampling', 6000)
            go('network', 700); pg.screenshot(path=str(shots / f'{tag}-3a-fadein.png'))
            pg.wait_for_timeout(4000); pg.screenshot(path=str(shots / f'{tag}-3-network.png'))
            r = pg.evaluate('__demo.ctx.state.reveal'); check(r['network.layers'] > 0.98 and r['forward.pulse'] < 0.01, f'{tag}: stage 3 reveal')
            info = pg.evaluate(f'{api}.getLayerInfo()'); check(len(info) == 6 and info[0]['shown'] == 3 and info[-1]['shown'] == 4 and info[1]['shown'] == 16 and info[1]['trueSize'] == 128, f'{tag}: layer info {len(info)} layers')
            go('forward', 1200); pg.screenshot(path=str(shots / f'{tag}-4a-flow-early.png'))
            pg.wait_for_timeout(1300); pg.screenshot(path=str(shots / f'{tag}-4b-flow-mid.png'))
            mid = pg.evaluate(f'{api}.state'); check(0 < mid['progress'] < 1 or mid['done'], f'{tag}: mid-pass progress {mid["progress"]:.2f}')
            pg.wait_for_timeout(5000); pg.screenshot(path=str(shots / f'{tag}-4-forward.png'))
            st = pg.evaluate(f'{api}.state'); check(st['done'] and pg.evaluate('__demo.ctx.modules.geometry.api.outputProgress') > 0.99, f'{tag}: pass completes, field resolved')
            pg.evaluate(f'{api}.replay()'); pg.wait_for_timeout(600); check(pg.evaluate(f'{api}.state.progress') < 0.5, f'{tag}: replay restarts'); pg.wait_for_timeout(5000)
            pg.evaluate(f'{api}.setPlaying(false)'); c0 = pg.evaluate(f'{api}.state.clock'); pg.wait_for_timeout(500); check(pg.evaluate(f'{api}.state.clock') == c0, f'{tag}: pause freezes')
            pg.evaluate(f'{api}.setPlaying(true)')
            base = None
            for rnd in range(2):
              for wd, dp in ((256, 6), (24, 2), (1024, 12), (16, 1), (128, 4), (512, 8), (64, 3)):
                  pg.evaluate(f"__demo.ctx.knobs.set('width', {wd}); __demo.ctx.knobs.set('depth', {dp})"); pg.wait_for_timeout(350)
                  li = pg.evaluate(f'{api}.getLayerInfo()'); check(len(li) == dp + 2 and li[1]['trueSize'] == wd and li[1]['shown'] == min(wd, 16), f'{tag}: rebuild {wd}x{dp}')
                  if wd == 1024: pg.screenshot(path=str(shots / f'{tag}-4c-w1024-d12.png')); pg.wait_for_timeout(1500); pg.screenshot(path=str(shots / f'{tag}-4d-w1024-d12.png'))
                  if wd == 24: pg.screenshot(path=str(shots / f'{tag}-4e-w24-d2.png'))
              for act in ('sin', 'silu', 'tanh'): pg.evaluate(f"__demo.ctx.knobs.set('activation', '{act}')"); pg.wait_for_timeout(250)
              pg.evaluate("__demo.ctx.knobs.set('width', 128); __demo.ctx.knobs.set('depth', 4)"); pg.wait_for_timeout(400)
              if rnd == 0: base = pg.evaluate(INFO)
            def _rebuild_ms(w, d): return pg.evaluate("([w, d]) => { const t = performance.now(); __demo.ctx.knobs.set('width', w); __demo.ctx.knobs.set('depth', d); return performance.now() - t; }", [w, d])
            ts = sorted(_rebuild_ms(w, d) for w, d in [(256, 5), (512, 9), (128, 4), (512, 9), (128, 6)]); ms = ts[len(ts) // 2]  # median of 5: robust to load spikes
            check(ms < 400, f'{tag}: rebuild median {ms:.0f} ms < 400 (software GL, machine-load tolerant)')
            pg.evaluate("__demo.ctx.knobs.set('width', 128); __demo.ctx.knobs.set('depth', 4)"); pg.wait_for_timeout(500)
            after = pg.evaluate(INFO); print('renderer.info', base, '->', after)
            check(after['g'] <= base['g'] and after['t'] <= base['t'], f'{tag}: no geometry/texture growth {base} -> {after}')
            check(after['calls'] < 90, f'{tag}: draw calls {after["calls"]} < 90')
            pg.evaluate("__demo.ctx.knobs.set('batch', 64)"); pg.wait_for_timeout(500); pg.evaluate("__demo.ctx.knobs.set('batch', 20000)"); pg.wait_for_timeout(800)
            pg.evaluate("__demo.ctx.knobs.set('batch', 4096)")
            pg.evaluate(f'{api}.setActivationHighlight(2, 1)'); pg.wait_for_timeout(300); pg.screenshot(path=str(shots / f'{tag}-4f-highlight.png')); pg.evaluate(f'{api}.clearHighlights()')
            go('memory-fwd', 3000)
            go('network', 3500); r = pg.evaluate('__demo.ctx.state.reveal'); check(r['forward.field'] < 0.02 and r['network.layers'] > 0.98, f'{tag}: reverse to stage 3 gates')
            go('geometry', 4500); r = pg.evaluate('__demo.ctx.state.reveal'); check(r['network.layers'] < 0.02 and r['network.edges'] < 0.02, f'{tag}: reverse to stage 1 hides net')
            pg.close()
        b.close()
finally:
    srv.terminate()
check(not errors, f'zero console errors {errors[:3]}')
print('NETWORK SMOKE', 'FAIL' if fails else 'PASS'); sys.exit(1 if fails else 0)
