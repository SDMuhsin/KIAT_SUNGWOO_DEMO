#!/usr/bin/env python3
"""Frame-time self-check across all slides (3D stages and 2D cards): python tests/perf.py [--seconds 3] [--url URL]. Uses window.__demo.perf().
Software GL => treat numbers as an UPPER bound on CPU cost; GPU cost is not measured."""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import serve, launch
from playwright.sync_api import sync_playwright
ap = argparse.ArgumentParser(); ap.add_argument('--seconds', type=float, default=3); ap.add_argument('--url'); ap.add_argument('--out'); a = ap.parse_args()
with sync_playwright() as p:
    b = launch(p); srv = None
    if a.url: url = a.url
    else: srv, port = serve(); url = f'http://127.0.0.1:{port}/index.html'
    pg = b.new_page(viewport={'width': 1280, 'height': 720})
    pg.goto(url); pg.wait_for_function('window.__demo && window.__demo.frames > 5', timeout=90000)
    rows = []
    print('stage  cpu_ms  update  render   max  calls   tris   points  geoms  tex')
    for i in range(pg.evaluate('__demo.STAGES.length')):
        pg.evaluate(f'__demo.controller.goTo({i})'); pg.wait_for_function('__demo.controller.settled', timeout=90000); pg.wait_for_timeout(2500)
        r = pg.evaluate(f'__demo.perf({a.seconds})'); rows.append(r)
        print(f"{r['stage']:>5} {r['cpuMs']:>7} {r['updateMs']:>7} {r['renderMs']:>7} {r['maxMs']:>6} {r['calls']:>6} {r['triangles']:>7} {r['points']:>8} {r['geometries']:>6} {r['textures']:>4}")
    if a.out: pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))
    if srv: srv.terminate()
    b.close()
