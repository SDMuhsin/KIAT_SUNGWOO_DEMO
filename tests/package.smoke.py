#!/usr/bin/env python3
"""Packaging smoke:  python tests/package.smoke.py [--quick]
 1. vendored/offline: serves the repo over http, ABORTS every non-local request, walks all 20 stages, zero console errors,
    zero attempted external requests.
 2. builds dist/sage-demo.html (tools/build_single.py) and loads it via file:// (no server), network blocked, walks all 20
    stages, zero console errors, WebGL canvas rendering, no <script type=module>/importmap in the file.
Works on Windows/macOS/Linux with `pip install playwright` + `playwright install chromium`."""
import argparse, subprocess, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import ROOT, serve, launch
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true'); ap.add_argument('--settle', type=float, default=0.6)
args = ap.parse_args()
fails = []
def check(c, m):
    print(('ok   ' if c else 'FAIL ') + m)
    if not c: fails.append(m)

subprocess.check_call([sys.executable, str(ROOT / 'tools' / 'build_single.py')])
dist = ROOT / 'dist' / 'sage-demo.html'
txt = dist.read_text(encoding='utf-8')
check('type="module"' not in txt and 'importmap' not in txt, 'dist: no module scripts / importmap')
check('cdn.jsdelivr' not in txt and 'unpkg' not in txt, 'dist: no CDN references')

def walk(b, url, tag):
    errors, external = [], []
    ctx = b.new_context(viewport={'width': 1280, 'height': 720})
    def gate(route):
        u = route.request.url
        if u.startswith(('http://127.0.0.1', 'http://localhost', 'file://', 'data:', 'blob:')): route.continue_()
        else: external.append(u); route.abort()
    ctx.route('**/*', gate)
    pg = ctx.new_page()
    pg.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
    pg.on('pageerror', lambda e: errors.append(str(e)))
    pg.on('requestfailed', lambda r: errors.append('requestfailed ' + r.url))
    pg.goto(url); pg.wait_for_function('window.__demo && window.__demo.frames > 5', timeout=90000)
    n = pg.evaluate('__demo.STAGES.length'); check(n == 20, f'{tag}: 20 slides')
    seen = []
    for i in range(n):
        if i: pg.keyboard.press('ArrowRight')
        pg.wait_for_timeout(int(args.settle * 1000))
        pg.wait_for_function('__demo.controller.settled', timeout=90000); seen.append(pg.evaluate('__demo.controller.index'))
        if args.quick and i > 4: break
    check(seen == list(range(len(seen))), f'{tag}: walked stages {seen[0]}..{seen[-1]}')
    check(pg.evaluate('__demo.renderer.info.render.calls') > 5, f'{tag}: renderer drawing')
    px = pg.evaluate("(() => { const c = document.createElement('canvas'); return !!document.getElementById('stage').getContext; })()")
    # --- presenter extras: help overlay, auto-tour, resize re-frame, perf self-check
    pg.keyboard.press('Home'); pg.wait_for_timeout(300)
    pg.keyboard.press('?'); pg.wait_for_timeout(150); check(pg.is_visible('#help'), f'{tag}: ? opens help overlay')
    pg.keyboard.press('Escape'); pg.wait_for_timeout(150); check(not pg.is_visible('#help'), f'{tag}: Esc closes help')
    pg.keyboard.press('h'); pg.wait_for_timeout(100); check(pg.is_visible('#help'), f'{tag}: H opens help'); pg.keyboard.press('h')
    pg.evaluate('__demo.extras.setTour(true)'); pg.wait_for_timeout(13500)
    check(pg.evaluate('__demo.controller.index') == 1, f'{tag}: auto-tour advanced one stage after ~12 s')
    pg.mouse.click(600, 300); pg.wait_for_timeout(200); check(pg.evaluate('__demo.extras.tour.paused'), f'{tag}: interaction pauses auto-tour')
    pg.evaluate('__demo.extras.setTour(false)')
    pg.evaluate('__demo.controller.goTo(3); __demo.controller.goTo(0)'); pg.wait_for_timeout(2500)
    c0 = pg.evaluate('__demo.ctx.camera.position.toArray()')
    pg.set_viewport_size({'width': 900, 'height': 700}); pg.wait_for_timeout(800)
    c1 = pg.evaluate('__demo.ctx.camera.position.toArray()')
    check(max(abs(a - b) for a, b in zip(c0, c1)) > 0.05, f'{tag}: camera re-frames on window resize {c0} -> {c1}')
    pf = pg.evaluate('__demo.perf(1)'); check(pf['frames'] > 0 and pf['cpuMs'] > 0 and pf['calls'] > 0, f'{tag}: __demo.perf() works {pf["cpuMs"]} ms')
    check(not external, f'{tag}: no external requests attempted {external[:3]}')
    check(not errors, f'{tag}: zero console errors {errors[:3]}')
    ctx.close()

try:
    with sync_playwright() as p:
        b = launch(p)
        srv, port = serve()
        try: walk(b, f'http://127.0.0.1:{port}/index.html', 'http+vendor')
        finally: srv.terminate()
        walk(b, dist.resolve().as_uri(), 'file://dist')
        b.close()
finally:
    pass
print('PACKAGE SMOKE', 'FAIL' if fails else 'PASS'); sys.exit(1 if fails else 0)
