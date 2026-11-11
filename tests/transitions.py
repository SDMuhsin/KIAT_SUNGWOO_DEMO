#!/usr/bin/env python3
"""Transition audit: python tests/transitions.py OUTDIR [--sizes desk,mobile] [--dirs fwd,rev] [--only a-b,...] [--url URL]
For every adjacent stage pair (ids from __demo.STAGES) jumps to A, navigates to B and saves a contact sheet
[just before | ~midpoint of the glide | settled] per pair and direction. Frame-driven waits (software GL is slow)."""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import serve, launch
from playwright.sync_api import sync_playwright
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--sizes', default='desk,mobile'); ap.add_argument('--dirs', default='fwd,rev')
ap.add_argument('--only', default=''); ap.add_argument('--url'); ap.add_argument('--mid', type=int, default=7); ap.add_argument('--end', type=int, default=45); a = ap.parse_args()
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
        ids = pg.evaluate('__demo.STAGES.map(s=>s.id)')
        def frames(n):
            f0 = pg.evaluate('__demo.frames'); pg.wait_for_function(f'__demo.frames >= {f0 + n}', timeout=120000)
        pairs = [(i, i + 1) for i in range(len(ids) - 1)]
        if a.only: want = set(a.only.split(',')); pairs = [q for q in pairs if f'{ids[q[0]]}-{ids[q[1]]}' in want or f'{q[0]+1}-{q[1]+1}' in want]
        for d in a.dirs.split(','):
            for (i, j) in pairs:
                s, t = (i, j) if d == 'fwd' else (j, i)
                pg.evaluate(f'__demo.controller.goTo({s}, {{instant:true}})'); frames(40)
                shots = []
                def snap(tag):
                    f = out / f'_{tag}.png'; pg.screenshot(path=str(f)); shots.append(Image.open(f).convert('RGB'))
                snap('a'); pg.evaluate(f'__demo.controller.goTo({t})'); frames(a.mid); snap('b'); frames(a.end - a.mid); snap('c')
                k = 0.5 if name == 'desk' else 0.6; ims = [im.resize((int(w * k), int(h * k))) for im in shots]
                sheet = Image.new('RGB', (sum(im.width for im in ims) + 8, ims[0].height), (60, 60, 60)); x = 0
                for im in ims: sheet.paste(im, (x, 0)); x += im.width + 4
                sheet.save(out / f'{name}-{d}-{s+1:02d}to{t+1:02d}.png'); print(name, d, ids[s], '->', ids[t], flush=True)
        pg.close()
    print('errors:', errs[:5])
    if srv: srv.terminate()
    b.close()
