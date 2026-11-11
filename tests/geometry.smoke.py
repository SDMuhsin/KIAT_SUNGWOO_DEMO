#!/usr/bin/env python3
"""Geometry scene smoke: python3 tests/geometry.smoke.py [--shots DIR]. Screenshots stages 1,2,4 (+violation view) at 1280x720 and 390x844,
asserts zero console errors, reversible gating, point-count knob, draw-call sanity (renderer.info)."""
import argparse, subprocess, sys, time, socket, os, pathlib, glob
from playwright.sync_api import sync_playwright
ROOT = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument('--shots', default=str(ROOT / 'llmdocs/shots/geometry')); a = ap.parse_args()
shots = pathlib.Path(a.shots); shots.mkdir(parents=True, exist_ok=True)
s = socket.socket(); s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]; s.close()
srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(port), '--bind', '127.0.0.1'], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(0.8)
errors, fails = [], []
def check(c, m):
    if not c: fails.append(m)
    print('ok  ' if c else 'FAIL', m)
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
            api = '__demo.ctx.modules.geometry.api'
            def go(stage, wait=4200):
                pg.evaluate(f"__demo.controller.goTo('{stage}')"); pg.wait_for_function('__demo.controller.settled', timeout=90000); pg.wait_for_timeout(wait)
            pg.wait_for_timeout(3000); pg.screenshot(path=str(shots / f'{tag}-1-geometry.png'))
            r = pg.evaluate('__demo.ctx.state.reveal'); check(r['sampling.interior'] < 0.01, f'{tag}: sampling hidden at stage 1')
            go('sampling', 1300); pg.screenshot(path=str(shots / f'{tag}-2a-emerge.png')); pg.wait_for_timeout(1700); pg.screenshot(path=str(shots / f'{tag}-2c-lifting.png')); pg.wait_for_timeout(3500); pg.screenshot(path=str(shots / f'{tag}-2-sampling.png'))
            check(pg.evaluate(f'{api}.pointCount') == 4096, f'{tag}: default 4096 points')
            pg.evaluate("__demo.ctx.knobs.set('batch', 12000)"); pg.wait_for_timeout(500); check(abs(pg.evaluate(f'{api}.pointCount') - 12000) < 64, f'{tag}: batch knob -> 12000 points')
            pg.evaluate("__demo.ctx.knobs.set('batch', 100000)"); pg.wait_for_timeout(500); check(pg.evaluate(f'{api}.pointCount') == 20000, f'{tag}: capped at 20000')
            pg.screenshot(path=str(shots / f'{tag}-2b-sampling-20k.png'))
            info = pg.evaluate('({calls: __demo.renderer.info.render.calls, tris: __demo.renderer.info.render.triangles, pts: __demo.renderer.info.render.points})'); print('renderer.info @20k', info)
            check(info['calls'] < 80, f'{tag}: draw calls {info["calls"]} < 80')
            pg.evaluate("__demo.ctx.knobs.set('batch', 4096)")
            go('forward', 5500); pg.screenshot(path=str(shots / f'{tag}-4-forward.png'))
            check(pg.evaluate(f'{api}.outputProgress') > 0.9, f'{tag}: output resolves on forward stage (auto)')
            go('pinn', 3500); pg.evaluate(f'{api}.setPhysicsViolation(0.9)'); pg.wait_for_timeout(1500); pg.screenshot(path=str(shots / f'{tag}-8-violation.png'))
            pg.evaluate(f'{api}.setPhysicsViolation(0)')
            # reverse
            go('geometry', 4500); r = pg.evaluate('__demo.ctx.state.reveal')
            check(r['sampling.interior'] < 0.02 and r['forward.field'] < 0.02 and r['geometry.duct'] > 0.98, f'{tag}: reverse to stage 1 gates correctly')
            pg.screenshot(path=str(shots / f'{tag}-1b-after-reverse.png'))
            pg.close()
        b.close()
finally:
    srv.terminate()
check(not errors, f'zero console errors {errors[:3]}')
print('GEOMETRY SMOKE', 'FAIL' if fails else 'PASS'); sys.exit(1 if fails else 0)
