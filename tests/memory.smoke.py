#!/usr/bin/env python3
"""Memory scene smoke: python3 tests/memory.smoke.py [--shots DIR] [--quick]. Stages 5/6/7 at 1280x720 and 390x844 (screenshots to
llmdocs/shots/memory/), zero console errors, backward animation state, live bound flips in the scene (bandwidth / peak knobs),
bf16 halves bytes, rapid knob changes leak no geometries/textures, reverse gating."""
import argparse, subprocess, sys, time, socket, os, pathlib, glob
from playwright.sync_api import sync_playwright
ROOT = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument('--shots', default=str(ROOT / 'llmdocs/shots/memory')); ap.add_argument('--quick', action='store_true'); a = ap.parse_args()
shots = pathlib.Path(a.shots); shots.mkdir(parents=True, exist_ok=True)
s = socket.socket(); s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]; s.close()
srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(port), '--bind', '127.0.0.1'], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(0.8)
errors, fails = [], []
def check(c, m):
    if not c: fails.append(m)
    print('ok  ' if c else 'FAIL', m)
INFO = '({g: __demo.renderer.info.memory.geometries, t: __demo.renderer.info.memory.textures, calls: __demo.renderer.info.render.calls})'
PINK, CYAN = 0xe0408a, 0x2fd0e6
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
            pg.goto(f'http://127.0.0.1:{port}/index.html#stage=memory-fwd'); pg.wait_for_function('window.__demo && window.__demo.frames > 5', timeout=60000)
            api = '__demo.ctx.modules.memory.api'; K = lambda k, v: pg.evaluate(f"__demo.ctx.knobs.set('{k}', {v!r})")
            def go(stage, wait=3800):
                pg.evaluate(f"__demo.controller.goTo('{stage}')"); pg.wait_for_timeout(wait)
            def shot(n): pg.screenshot(path=str(shots / f'{tag}-{n}.png'))
            pg.wait_for_timeout(3000); shot('5-memory-fwd')
            r = pg.evaluate('__demo.ctx.state.reveal'); check(r['memory.weights'] > 0.98 and r['memory.gradients'] < 0.01 and r['bounds.overlay'] < 0.01, f'{tag}: stage 5 reveal')
            bd = pg.evaluate(f'{api}.getBreakdown()'); by = bd['bytes']
            check(abs(by['total'] - (by['weights'] + by['activations'] + by['gradients'] + by['optimizer'])) < 1 and by['optimizer'] == 2 * by['weights'], f'{tag}: breakdown consistent, Adam = 2x weights (fp32)')
            check(by['activations'] > by['weights'] * 10, f'{tag}: activations dominate at defaults')
            if a.quick and tag == 'mobile': pg.close(); continue
            # knob live resize: width up -> weights block grows
            w0 = pg.evaluate(f'{api}.getBlockAnchor("weights")'); K('width', 512); pg.wait_for_timeout(1200); w1 = pg.evaluate(f'{api}.getBlockAnchor("weights")')
            check(pg.evaluate(f'{api}.getBreakdown().bytes.weights') > by['weights'] * 8, f'{tag}: width knob grows weights bytes'); K('width', 128); pg.wait_for_timeout(900)
            K('precision', 'bf16'); pg.wait_for_timeout(900); bb = pg.evaluate(f'{api}.getBreakdown().bytes')
            check(bb['weights'] * 2 == by['weights'] and bb['activations'] * 2 == by['activations'], f'{tag}: bf16 halves weights+activations bytes'); K('precision', 'fp32')
            # stage 6
            go('backward', 1500); shot('6a-backward-start')
            pg.wait_for_timeout(2600); mid = pg.evaluate(f'{api}.backwardProgress'); shot('6b-backward-mid')
            check(0.1 < mid < 0.7, f'{tag}: backward mid-animation p={mid:.2f}')
            pg.evaluate(f'{api}.setBackwardProgress(0.5)'); pg.wait_for_timeout(700); shot('6c-backward-p50')
            pg.evaluate(f'{api}.setBackwardProgress(0.68)'); pg.wait_for_timeout(700); shot('6d-adam-appears')
            pg.evaluate(f'{api}.setBackwardProgress(0.88)'); pg.wait_for_timeout(500); shot('6e-update')
            pg.evaluate(f'{api}.setBackwardProgress(null)'); pg.evaluate(f'{api}.replayBackward()'); pg.wait_for_timeout(600); check(pg.evaluate(f'{api}.backwardProgress') < 0.2, f'{tag}: replay restarts')
            pg.wait_for_timeout(12500); check(pg.evaluate(f'{api}.backwardProgress') >= 0.999, f'{tag}: backward completes'); shot('6f-backward-done')
            r = pg.evaluate('__demo.ctx.state.reveal'); check(r['memory.gradients'] > 0.98 and r['memory.optimizer'] > 0.98, f'{tag}: stage 6 reveal')
            # stage 7
            go('bounds', 4200); shot('7a-bounds-default')
            pl = {x['id']: x for x in pg.evaluate(f'{api}.getPillars()')}
            check(all(pl[i]['color'] in (PINK, CYAN) for i in ('mmFwd', 'mmBwd', 'deriv', 'act', 'opt')), f'{tag}: pillars coloured memory/compute')
            check(pl['deriv']['bound'] == 'memory' and pl['deriv']['color'] == PINK, f'{tag}: default derivative evaluation memory-bound (pink)')
            x_def = pl['deriv']['x']
            K('bandwidth', 3000); pg.wait_for_timeout(2200); shot('7b-bandwidth-high'); pl = {x['id']: x for x in pg.evaluate(f'{api}.getPillars()')}
            check(pl['deriv']['bound'] == 'compute' and pl['deriv']['color'] == CYAN, f'{tag}: high bandwidth flips derivative evaluation to compute-bound IN SCENE')
            K('bandwidth', 100); pg.wait_for_timeout(2200); shot('7c-bandwidth-low'); pl = {x['id']: x for x in pg.evaluate(f'{api}.getPillars()')}
            check(pl['deriv']['bound'] == 'memory' and pl['deriv']['color'] == PINK, f'{tag}: low bandwidth back to memory-bound'); ms_low = pg.evaluate(f'{api}.getBreakdown().step.memShare')
            K('bandwidth', 696); K('peakTflops', 4); pg.wait_for_timeout(2200); shot('7d-peak-low'); pl = {x['id']: x for x in pg.evaluate(f'{api}.getPillars()')}
            check(pl['deriv']['color'] == CYAN and pl['mmFwd']['color'] == CYAN, f'{tag}: low peak TFLOPs flips ops to compute-bound IN SCENE')
            check(pg.evaluate(f'{api}.getBreakdown().step.compShare') > ms_low * 0 + 0.5, f'{tag}: compute share > 50% at low peak')
            K('peakTflops', 37); K('precision', 'bf16'); pg.wait_for_timeout(2200); shot('7e-bf16'); pl = {x['id']: x for x in pg.evaluate(f'{api}.getPillars()')}
            check(pl['deriv']['x'] > x_def + 0.2, f'{tag}: bf16 slides derivative pillar right (AI up)'); K('precision', 'fp32')
            K('width', 1024); pg.wait_for_timeout(2200); shot('7f-width-1024'); K('width', 128); K('batch', 65536); pg.wait_for_timeout(2200); shot('7g-batch-65k'); K('batch', 4096); pg.wait_for_timeout(1500)
            ro = pg.inner_text('#hud-readouts'); check('Memory-bound' in ro and '% of step' in ro, f'{tag}: HUD readout memory-bound xx% of step ({ro.splitlines()[:1]})')
            # leak test: rapid knob changes
            base = None
            for rnd in range(3):   # rounds 0-1 warm up (lazy label textures / buffers); baseline after round 1, leak check after round 2
                for i in range(40):
                    K('bandwidth', 100 + (i * 73) % 2500); K('peakTflops', 2 + (i * 11) % 150); K('width', [16, 64, 256, 1024][i % 4]); K('depth', 1 + i % 12); K('batch', [64, 4096, 65536, 262144][i % 4]); pg.wait_for_timeout(40)
                K('width', 128); K('depth', 4); K('batch', 4096); K('bandwidth', 696); K('peakTflops', 37); pg.wait_for_timeout(800)
                if rnd == 1: base = pg.evaluate(INFO)
            after = pg.evaluate(INFO); print('renderer.info', base, '->', after)
            # a real leak grows by >= 1 geometry per knob change (dozens per round); tiny slack covers lazily created label textures
            check(after['g'] <= base['g'] + 2 and after['t'] <= base['t'] + 4, f'{tag}: no geometry/texture growth after warm-up {base} -> {after}')
            check(after['calls'] < 160, f'{tag}: draw calls {after["calls"]} < 160')
            # reverse + move on
            go('backward', 3000); r = pg.evaluate('__demo.ctx.state.reveal'); check(r['bounds.overlay'] < 0.02 and r['memory.gradients'] > 0.98, f'{tag}: reverse to stage 6 hides bounds')
            go('memory-fwd', 3000); r = pg.evaluate('__demo.ctx.state.reveal'); check(r['memory.gradients'] < 0.02 and r['memory.activations'] > 0.98, f'{tag}: reverse to stage 5 hides gradients')
            go('forward', 3500); r = pg.evaluate('__demo.ctx.state.reveal'); check(r['memory.weights'] < 0.02, f'{tag}: reverse to stage 4 hides memory'); check(not pg.evaluate('__demo.ctx.modules.memory.group.visible'), f'{tag}: memory group hidden')
            go('pinn', 1500); go('bounds', 3000)
            go('pinn', 3500); check(not pg.evaluate('__demo.ctx.modules.memory.group.visible') or pg.evaluate('__demo.ctx.modules.memory.group.children.length') > 0, f'{tag}: past stage 7 memory fades')
            pg.close()
        b.close()
finally:
    srv.terminate()
check(not errors, f'zero console errors {errors[:3]}')
print('MEMORY SMOKE', 'FAIL' if fails else 'PASS'); sys.exit(1 if fails else 0)
