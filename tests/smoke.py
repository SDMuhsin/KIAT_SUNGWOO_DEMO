#!/usr/bin/env python3
"""End-to-end smoke test. Run from repo root:  python3 tests/smoke.py [--shots DIR] [--tag NAME]
Needs `pip install playwright` + a chromium (`playwright install chromium`). Serves the repo via http.server,
walks all stages forward and back, checks reveal gating, knobs, hash deep-link, keys, no overflow, zero console errors."""
import argparse, subprocess, sys, time, socket, os, pathlib
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument('--shots', default=str(ROOT / 'llmdocs/shots/foundation')); ap.add_argument('--settle', type=float, default=2.2)
args = ap.parse_args(); shots = pathlib.Path(args.shots); shots.mkdir(parents=True, exist_ok=True)

s = socket.socket(); s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]; s.close()
srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(port), '--bind', '127.0.0.1'], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.8)
errors, fails = [], []
def check(cond, msg):
    if not cond: fails.append(msg); print('FAIL', msg)
    else: print('ok  ', msg)

try:
    with sync_playwright() as p:
        import glob
        la = dict(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
        try: b = p.chromium.launch(**la)
        except Exception:  # playwright/browser version mismatch: use any installed chromium (or $CHROMIUM_PATH)
            cands = [os.environ.get('CHROMIUM_PATH')] + sorted(glob.glob(os.path.expanduser('~/.cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell')) + glob.glob(os.path.expanduser('~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome')))
            b = p.chromium.launch(executable_path=next(c for c in cands if c), **la)
        def newpage(w, h):
            pg = b.new_page(viewport={'width': w, 'height': h})
            _wt = pg.wait_for_timeout; pg.wait_for_timeout = lambda ms: _wt(int(ms * float(os.environ.get('SMOKE_SLOW', '2'))))  # animations are frame-driven; software GL is slow, so scale fixed waits (SMOKE_SLOW=1 to disable)
            pg.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
            pg.on('pageerror', lambda e: errors.append(str(e)))
            pg.on('requestfailed', lambda r: errors.append('requestfailed ' + r.url))
            return pg
        pg = newpage(1280, 720); url = f'http://127.0.0.1:{port}/index.html'
        pg.goto(url); pg.wait_for_function('window.__demo && window.__demo.frames > 5', timeout=60000)
        n = pg.evaluate('__demo.STAGES.length'); check(n == 15, f'15 stages (got {n})')
        parts_by_stage = pg.evaluate('__demo.STAGES.map(s=>s.parts)')
        def reveal_ok(i, tag):
            r = pg.evaluate('__demo.ctx.state.reveal'); bad = []
            for k, parts in enumerate(parts_by_stage):
                for part in parts:
                    want = 1 if k <= i else 0
                    if abs(r[part] - want) > 0.02: bad.append((part, r[part], want))
            check(not bad, f'{tag}: reveal gating at stage {i} {bad[:2]}')
        def idx(): return pg.evaluate('__demo.ctx.state.stageIndex')
        pg.wait_for_timeout(int(args.settle * 1000)); reveal_ok(0, 'initial')
        pg.screenshot(path=str(shots / 'fwd-01.png'))
        for i in range(1, n):
            pg.keyboard.press('ArrowRight' if i % 2 else 'Space') if i < 3 else pg.click('#btn-next')
            pg.wait_for_timeout(int(args.settle * 1000)); check(idx() == i, f'forward -> {i}')
            reveal_ok(i, 'fwd'); pg.screenshot(path=str(shots / f'fwd-{i+1:02d}.png'))
            check(pg.inner_text('#hud-count') == f'{i+1} / {n}', f'progress text {i+1}')
        check(pg.is_disabled('#btn-next'), 'next disabled at end')
        for i in range(n - 2, -1, -1):
            pg.keyboard.press('ArrowLeft'); pg.wait_for_timeout(1800); check(idx() == i, f'back -> {i}')
            if i in (13, 7, 0): reveal_ok(i, 'back'); pg.screenshot(path=str(shots / f'back-{i+1:02d}.png'))
        # dots + hash
        pg.click('#hud-dots button:nth-child(7)'); pg.wait_for_timeout(300); check(idx() == 6 and pg.evaluate('location.hash') == '#stage=bounds', 'dot click sets stage + hash')
        # knob panel: bounds stage shows 6 knobs; moving slider updates knob + live readout
        check(pg.locator('#knob-panel .knob').count() == 6, 'bounds stage shows 6 knobs')
        before = pg.evaluate("__demo.ctx.knobs.get('width')"); pg.locator('#knob-width').fill('900'); pg.wait_for_timeout(200)
        after = pg.evaluate("__demo.ctx.knobs.get('width')"); check(after > before, f'width slider live ({before}->{after})')
        pg.select_option('#knob-precision', 'bf16'); check(pg.evaluate("__demo.ctx.knobs.get('precision')") == 'bf16', 'precision select')
        check('model estimate' in pg.inner_text('#hud-readouts'), 'model-estimate readout shown')
        check(pg.locator('#hud-badges .badge').count() >= 1, 'badges shown')
        pg.close(); pg = newpage(1280, 720)
        pg.goto(url + '#stage=sage-jax'); pg.wait_for_function('window.__demo && window.__demo.frames > 5'); pg.wait_for_timeout(500)
        check(idx() == 13, 'hash deep link -> sage-jax'); reveal_ok(13, 'deeplink')
        check(pg.locator('#knob-sageHalf').count() == 1, 'toggle knob rendered')
        for w, h, tag in ((1280, 720, 'desk'), (390, 844, 'mobile')):
            q = newpage(w, h); q.goto(url + '#stage=backward'); q.wait_for_function('window.__demo && window.__demo.frames > 5'); q.wait_for_timeout(800)
            ov = q.evaluate('[document.documentElement.scrollWidth - innerWidth, document.documentElement.scrollHeight - innerHeight]')
            check(ov[0] <= 0 and ov[1] <= 0, f'no page overflow at {w}x{h} {ov}')
            hb = q.evaluate("(()=>{const a=document.getElementById('hud-top').getBoundingClientRect(),c=document.getElementById('hud-bottom').getBoundingClientRect();return c.top>=a.bottom-1&&c.left>=0&&c.right<=innerWidth})()")
            check(hb, f'top/bottom HUD cards do not collide at {w}x{h}')
            q.screenshot(path=str(shots / f'{tag}-backward.png')); q.close()
        b.close()
finally:
    srv.terminate()
real = [e for e in errors]
check(not real, f'zero console errors ({real[:3]})')
print('SMOKE', 'FAIL' if fails else 'PASS', f'({len(fails)} failures)'); sys.exit(1 if fails else 0)
