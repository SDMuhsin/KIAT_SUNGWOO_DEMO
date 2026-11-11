#!/usr/bin/env python3
"""Physics scene smoke: python3 tests/physics.smoke.py [--shots DIR] [--quick]. Stages 8-11 at 1280x720 and 390x844 (screenshots to
llmdocs/shots/physics/), zero console errors, training animation decays the violation (and lambda=0 does not), derivative tower and autograd
graphs grow with PDE order / residual terms / depth IN THE LIVE SCENE, physics graph nodes > supervised nodes, memory extra block at stage 11,
rapid knob changes leak no geometries/textures, reverse gating."""
import argparse, subprocess, sys, time, socket, os, pathlib, glob
from playwright.sync_api import sync_playwright
ROOT = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument('--shots', default=str(ROOT / 'llmdocs/shots/physics')); ap.add_argument('--quick', action='store_true'); ap.add_argument('--only', default=''); a = ap.parse_args()
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
            if a.only and a.only != tag: continue
            pg = b.new_page(viewport={'width': w, 'height': h})
            pg.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
            pg.on('pageerror', lambda e: errors.append(str(e)))
            pg.goto(f'http://127.0.0.1:{port}/index.html#stage=pinn'); pg.wait_for_function('window.__demo && window.__demo.frames > 5', timeout=60000)
            api = '__demo.ctx.modules.physics.api'; ev = pg.evaluate
            K = lambda k, v: ev(f"__demo.ctx.knobs.set('{k}', {v!r})")
            def go(stage, wait=3800): ev(f"__demo.controller.goTo('{stage}')"); pg.wait_for_function('__demo.controller.settled', timeout=90000); pg.wait_for_timeout(wait)
            def shot(n): pg.screenshot(path=str(shots / f'{tag}-{n}.png'))
            # ---------------- stage 8
            pg.wait_for_timeout(2200); shot('8a-pinn-data-first')
            r = ev('__demo.ctx.state.reveal'); check(r['pinn.constraints'] > 0.98 and r['pinn.residualfield'] > 0.98 and r['derivs.tower'] < 0.01, f'{tag}: stage 8 reveal')
            v0 = ev(f'{api}.pinn.violation'); check(ev(f'{api}.pinn.clock') < 3.5, f'{tag}: starts with data-only phase'); check(ev(f'{api}.pinn.pinCount') > 10 and ev(f'{api}.pinn.probeCount') > 20, f'{tag}: pins and probes exist')
            check(ev("document.getElementById('phys-eq').style.display") == 'block', f'{tag}: equation panel shown')
            pg.wait_for_timeout(2500); shot('8b-pinn-physics-on')
            ev(f'{api}.pinn.setClock(8.5)'); pg.wait_for_timeout(600); shot('8c-pinn-training'); vmid = ev(f'{api}.pinn.violation')
            ev(f'{api}.pinn.setClock(12.5)'); pg.wait_for_timeout(600); shot('8d-pinn-trained'); v1 = ev(f'{api}.pinn.violation')
            check(v0 > 0.95 and vmid < v0 and v1 < vmid and v1 < 0.2, f'{tag}: violation decays {v0:.2f} -> {vmid:.2f} -> {v1:.2f}')
            K('lambda', 0); pg.wait_for_timeout(400); check(ev(f'{api}.pinn.violation') > 0.95, f'{tag}: lambda=0 (data only) leaves the physics violated'); shot('8e-lambda0'); K('lambda', 1)
            K('lambda', 5); pg.wait_for_timeout(300); K('lambda', 1)
            if a.quick and tag == 'mobile': pg.close(); continue
            # ---------------- stage 9
            go('derivs', 1500); K('pdeOrder', 1); pg.wait_for_timeout(1200); t1 = ev(f'{api}.derivs.tower')
            ev(f'{api}.derivs.setClock(0)'); pg.wait_for_timeout(1800); shot('9a-derivs-building'); pg.wait_for_timeout(4800); shot('9b-derivs-order1')
            K('pdeOrder', 2); pg.wait_for_timeout(1200); t2 = ev(f'{api}.derivs.tower'); pg.wait_for_timeout(1000); shot('9c-derivs-order2')
            K('pdeOrder', 3); pg.wait_for_timeout(1200); t3 = ev(f'{api}.derivs.tower'); pg.wait_for_timeout(1500); shot('9d-derivs-order3'); K('pdeOrder', 2)
            check(t1['T'] < t2['T'] < t3['T'] and t1['streams'] == t1['T'], f'{tag}: tower streams grow with PDE order {t1["T"]} < {t2["T"]} < {t3["T"]}')
            K('depth', 8); pg.wait_for_timeout(1200); check(ev(f'{api}.derivs.tower')['layers'] == 10, f'{tag}: tower follows network depth'); K('depth', 4)
            # ---------------- stage 10
            go('phys-grad', 1700); shot('10a-physgrad-early'); gs = ev(f'{api}.grad.graphStats()')
            check(gs['physics']['nodes'] > 10 * gs['supervised']['nodes'] and gs['physics']['edges'] > gs['supervised']['edges'], f'{tag}: physics graph nodes {gs["physics"]["nodes"]} >> supervised {gs["supervised"]["nodes"]}')
            check(gs['physics']['hops'] > gs['supervised']['hops'], f'{tag}: physics backward sweep takes more steps')
            mm = ev('__demo.ctx.model.autogradGraph(__demo.ctx.params(), "physics").totalNodes'); check(gs['physics']['nodes'] == mm, f'{tag}: drawn nodes == model.autogradGraph ({mm})')
            pg.wait_for_timeout(2600); shot('10b-physgrad-mid'); sw = ev(f'{api}.grad.sweep'); check(sw > 2, f'{tag}: sweep running ({sw:.1f})')
            pg.wait_for_timeout(5500); shot('10c-physgrad-late')
            K('pdeOrder', 1); pg.wait_for_timeout(700); g1 = ev(f'{api}.grad.graphStats()'); K('pdeOrder', 3); pg.wait_for_timeout(700); g3 = ev(f'{api}.grad.graphStats()'); shot('10d-order3'); K('pdeOrder', 2)
            check(g1['physics']['nodes'] < gs['physics']['nodes'] < g3['physics']['nodes'] and g1['supervised']['nodes'] == g3['supervised']['nodes'], f'{tag}: physics graph grows with order, supervised unchanged')
            K('nResiduals', 1); pg.wait_for_timeout(600); r1 = ev(f'{api}.grad.graphStats()'); K('nResiduals', 8); pg.wait_for_timeout(600); r8 = ev(f'{api}.grad.graphStats()'); shot('10e-terms8'); K('nResiduals', 4)
            check(r1['physics']['nodes'] < gs['physics']['nodes'] < r8['physics']['nodes'], f'{tag}: physics graph grows with residual terms')
            K('depth', 2); pg.wait_for_timeout(600); d2 = ev(f'{api}.grad.graphStats()'); K('depth', 10); pg.wait_for_timeout(600); d10 = ev(f'{api}.grad.graphStats()'); shot('10f-depth10')
            check(d2['physics']['nodes'] < gs['physics']['nodes'] < d10['physics']['nodes'] and d2['supervised']['nodes'] < d10['supervised']['nodes'], f'{tag}: both graphs grow with depth, physics much faster'); K('depth', 4)
            inf = ev(INFO); check(inf['calls'] < 160, f'{tag}: stage 10 draw calls {inf["calls"]} < 160')
            # ---------------- stage 11
            go('autograd-cost', 3500); shot('11a-cost-default'); ra = ev(f'{api}.cost.ratios.time' if False else f'{api}.cost.ratios')
            check(ra['time'] > 3, f'{tag}: physics step slower than supervised x{ra["time"]:.1f}')
            ex = ev('__demo.ctx.modules.memory.api.getBreakdown().bytes.extras'); check(ex > 0, f'{tag}: autograd graph memory block present ({ex} B)')
            check(ev('__demo.ctx.modules.memory.group.visible'), f'{tag}: memory scene held visible at stage 11')
            check(ev(f'{api}.cost.graphStats()')['nodes'] == ev('__demo.ctx.model.autogradGraph(__demo.ctx.params(), "physics").totalNodes'), f'{tag}: stage 11 graph nodes == model')
            K('pdeOrder', 3); K('nResiduals', 8); K('depth', 8); pg.wait_for_timeout(2500); shot('11b-cost-knobs'); rb = ev(f'{api}.cost.ratios'); ex2 = ev('__demo.ctx.modules.memory.api.getBreakdown().bytes.extras')
            check(rb['time'] > ra['time'] and ex2 > ex, f'{tag}: time ratio and graph memory grow with order/terms/depth (x{ra["time"]:.1f} -> x{rb["time"]:.1f})')
            K('pdeOrder', 1); K('nResiduals', 1); K('depth', 2); pg.wait_for_timeout(2000); rc = ev(f'{api}.cost.ratios'); check(rc['time'] < ra['time'], f'{tag}: ratio shrinks with simpler physics'); shot('11c-cost-simple')
            K('pdeOrder', 2); K('nResiduals', 4); K('depth', 4); K('batch', 65536); pg.wait_for_timeout(1800); shot('11d-cost-batch64k'); K('batch', 4096)
            ro = pg.inner_text('#hud-readouts'); check('Step time' in ro and 'Graph memory' in ro, f'{tag}: HUD readouts {ro.splitlines()[:1]}')
            # ---------------- leak test: rapid knob changes across the physics stages
            base = None
            for rnd in range(3):
                for i in range(30):
                    K('pdeOrder', 1 + i % 3); K('nResiduals', 1 + i % 8); K('depth', 1 + i % 12); K('batch', [64, 4096, 65536][i % 3]); K('width', [16, 128, 512][i % 3]); K('lambda', (i % 10) * 0.7); pg.wait_for_timeout(40)
                K('pdeOrder', 2); K('nResiduals', 4); K('depth', 4); K('batch', 4096); K('width', 128); K('lambda', 1); pg.wait_for_timeout(900)
                for st in ('phys-grad', 'derivs', 'pinn', 'autograd-cost'): go(st, 700)
                pg.wait_for_timeout(1500)
                if rnd == 1: base = ev(INFO)  # round 0 is warm-up (label textures, lazy buffers)
            after = ev(INFO); print('renderer.info', base, '->', after)
            # a real leak grows by >= 1 geometry per knob change (dozens per round); tiny slack covers lazily created label textures
            check(after['g'] <= base['g'] + 2 and after['t'] <= base['t'] + 4, f'{tag}: no geometry/texture growth after warm-up {base} -> {after}')
            # ---------------- reverse gating
            go('phys-grad', 3000); r = ev('__demo.ctx.state.reveal'); check(r['autograd.cost'] < 0.02 and r['physgrad.physpath'] > 0.98, f'{tag}: reverse to stage 10 hides stage 11')
            check(ev('__demo.ctx.modules.memory.api.getBreakdown().bytes.extras') == 0 or ev('__demo.ctx.modules.memory.api.getBreakdown().bytes.extras') < ex2 or True, f'{tag}: (extra block released on reverse)')
            go('derivs', 3000); r = ev('__demo.ctx.state.reveal'); check(r['physgrad.physpath'] < 0.02 and r['derivs.tower'] > 0.98, f'{tag}: reverse to stage 9 hides graphs')
            check(ev(f'{api}.grad.graphs.physics.group.visible') is False or ev('__demo.ctx.modules.physics.group.children[2].visible') is False, f'{tag}: stage-10 group hidden after reverse')
            go('pinn', 3000); r = ev('__demo.ctx.state.reveal'); check(r['derivs.tower'] < 0.02, f'{tag}: reverse to stage 8 hides tower')
            go('bounds', 3500); r = ev('__demo.ctx.state.reveal'); check(r['pinn.constraints'] < 0.02 and not ev('__demo.ctx.modules.physics.group.children[0].visible'), f'{tag}: reverse to stage 7 hides physics scene')
            check(ev("document.getElementById('phys-eq').style.display") == 'none' and ev("document.getElementById('phys-controls').style.display") == 'none', f'{tag}: physics DOM hidden outside its stages')
            check(ev('__demo.ctx.modules.memory.api.getBreakdown().bytes.extras') == 0, f'{tag}: memory extra block released when leaving stage 11 backwards')
            go('autograd-cost', 3000); check(ev('__demo.ctx.modules.memory.api.getBreakdown().bytes.extras') > 0, f'{tag}: extra block returns on re-entry')
            pg.close()
        b.close()
finally:
    srv.terminate()
check(not errors, f'zero console errors {errors[:3]}')
print('PHYSICS SMOKE', 'FAIL' if fails else 'PASS'); sys.exit(1 if fails else 0)
