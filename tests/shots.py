#!/usr/bin/env python3
"""Screenshot helper: python tests/shots.py OUTDIR [--stages 1,4,7] [--url URL] [--settle S].
Captures each requested stage at 1280x720 (desk) and 390x844 (mobile) via the #stage=ID deep link."""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import ROOT, serve, launch
from playwright.sync_api import sync_playwright
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--stages', default='1,4,8,10,13,14,16,18,20'); ap.add_argument('--settle', type=float, default=4.5)
ap.add_argument('--url'); ap.add_argument('--sizes', default='desk,mobile'); a = ap.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
SZ = {'desk': (1280, 720), 'mobile': (390, 844)}
with sync_playwright() as p:
    b = launch(p); srv = None
    if a.url: base = a.url
    else: srv, port = serve(); base = f'http://127.0.0.1:{port}/index.html'
    errs = []
    for name in a.sizes.split(','):
        w, h = SZ[name]; pg = b.new_page(viewport={'width': w, 'height': h})
        pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None); pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(base); pg.wait_for_function('window.__demo && window.__demo.frames > 5', timeout=90000)
        for s in a.stages.split(','):
            i = int(s)
            pg.evaluate(f'__demo.controller.goTo({i - 1})'); pg.wait_for_function('__demo.controller.settled', timeout=90000); pg.wait_for_timeout(int(a.settle * 1000))
            pg.screenshot(path=str(out / f'{name}-{i:02d}.png'))
        pg.close()
    print('errors:', errs[:5])
    if srv: srv.terminate()
    b.close()
