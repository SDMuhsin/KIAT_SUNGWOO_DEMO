#!/usr/bin/env python3
"""SAGE scene smoke: python3 tests/sage.smoke.py [--shots DIR] [--only desk|mobile]. Stages 12-15 at 1280x720 and 390x844 (screenshots to
llmdocs/shots/sage/), zero console errors, tape/pipeline phases, the stage-13 blend collapses the physics graph and MONOTONICALLY reduces AD-graph
memory / nodes / matmuls / step time (also in the memory scene's extra block), stage-14 toggles + 'measured' only for the deck figures,
stage-15 numbers equal the SPEC facts (no 90x outside the correction footnote), Amdahl slider, rapid knob changes leak nothing, reverse gating."""
import argparse, json, subprocess, sys, time, socket, os, pathlib, glob
from playwright.sync_api import sync_playwright
ROOT = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument('--shots', default=str(ROOT / 'llmdocs/shots/sage')); ap.add_argument('--only', default=''); a = ap.parse_args()
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
            _wt = pg.wait_for_timeout; pg.wait_for_timeout = lambda ms: _wt(int(ms * float(os.environ.get('SMOKE_SLOW', '2'))))  # animations are frame-driven; software GL is slow, so scale fixed waits (SMOKE_SLOW=1 to disable)
            pg.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
            pg.on('pageerror', lambda e: errors.append(str(e)))
            pg.goto(f'http://127.0.0.1:{port}/index.html#stage=sage-idea'); pg.wait_for_function('window.__demo && window.__demo.frames > 5', timeout=60000)
            ev = pg.evaluate; api = '__demo.ctx.modules.sage.api'
            K = lambda k, v: ev(f"__demo.ctx.knobs.set('{k}', {json.dumps(v)})")
            def go(stage, wait=3800): ev(f"__demo.controller.goTo('{stage}')"); pg.wait_for_timeout(wait)
            def shot(n): pg.screenshot(path=str(shots / f'{tag}-{n}.png'))
            EXTRA = '__demo.ctx.modules.memory.api.getBreakdown().bytes.extras'
            # ---------------- stage 12
            pg.wait_for_timeout(1500)
            r = ev('__demo.ctx.state.reveal'); check(r['sage.split'] > 0.98 and r['sage.tape'] > 0.98 and r['sage.before_after'] < 0.01, f'{tag}: stage 12 reveal')
            check(ev(f'{api}.idea.tapeLength') == 40 and ev(f'{api}.idea.ruleCount') == 7, f'{tag}: 40-entry tape, 7 VJP rule chips')
            check(ev("document.getElementById('sage-console').style.display") == 'block' and 'Prerequisite' in ev("document.getElementById('sage-console').textContent"), f'{tag}: console + prerequisite caveat shown')
            for name, t, ph in (('12a-split', 2.0, 'split'), ('12b-trace', 5.4, 'trace'), ('12c-reverse', 10.5, 'reverse'), ('12d-emit', 16.0, 'emit'), ('12e-compile', 18.8, 'compile'), ('12f-loop', 24.0, 'loop')):
                ev(f'{api}.idea.setClock({t})'); pg.wait_for_timeout(500); shot(name)
                check(ev(f'{api}.idea.phase') == ph, f'{tag}: phase {ph} at t={t}')
                txt = ev(f'{api}.idea.consoleText')
                if ph == 'trace': check('[' in txt and '@' in txt or '*' in txt, f'{tag}: tape lines shown ({txt.splitlines()[-1][:40]!r})')
                if ph == 'emit': check('def generated_backward' in txt, f'{tag}: emitted code shown')
                if ph == 'loop': check('called' in txt, f'{tag}: training loop calls the cached fn')
            check(ev(f'{api}.idea.state.calls') > 0, f'{tag}: calls counter ran')
            # ---------------- stage 13 : blend collapse (monotonic, in the live scene)
            go('sage-overhead', 2600)
            check(ev('__demo.ctx.state.reveal')['sage.before_after'] > 0.98, f'{tag}: stage 13 reveal')
            K('sageBlend', 0); pg.wait_for_timeout(700); shot('13a-blend0')
            series = []
            for bl in (0, 0.25, 0.5, 0.75, 1.0):
                K('sageBlend', bl); pg.wait_for_timeout(900)
                o = ev(f'({{st: {api}.overhead.state, col: {api}.overhead.before.collapse, vis: {api}.overhead.before.group.visible, ex: {EXTRA}, nodes: {api}.overhead.before.stats.nodes}})')
                series.append((bl, o))
                if bl == 0.5: shot('13b-blend0.5')
                if bl == 1.0: shot('13c-blend1')
            mono = lambda f: all(f(series[i + 1][1]) <= f(series[i][1]) + 1e-9 for i in range(len(series) - 1))
            check(all(abs(o['col'] - bl) < 1e-6 for bl, o in series), f'{tag}: graph.setCollapse follows the blend')
            check(mono(lambda o: o['st']['memBytes']) and series[0][1]['st']['memBytes'] > 1e6 and series[-1][1]['st']['memBytes'] == 0, f'{tag}: AD-graph memory shrinks monotonically to 0 ({series[0][1]["st"]["memBytes"]:.0f} -> 0)')
            check(mono(lambda o: o['ex']) and series[0][1]['ex'] > 0 and series[-1][1]['ex'] == 0, f'{tag}: memory scene extra block follows ({series[0][1]["ex"]} -> {series[-1][1]["ex"]})')
            check(mono(lambda o: o['st']['total']) and series[-1][1]['st']['total'] < series[0][1]['st']['total'] / 5, f'{tag}: step time shrinks monotonically ({series[0][1]["st"]["total"]*1e3:.2f} -> {series[-1][1]["st"]["total"]*1e3:.2f} ms)')
            check(mono(lambda o: o['st']['adNodes']) and mono(lambda o: o['st']['matmuls']) and series[0][1]['st']['matmuls'] == 24 and series[-1][1]['st']['matmuls'] == 10, f'{tag}: nodes fall, matmuls 24 -> 10')
            check(series[0][1]['vis'] and not series[-1][1]['vis'], f'{tag}: physics graph visible at 0 and gone at 1')
            n0 = series[0][1]['nodes']; check(n0 == ev('__demo.ctx.model.autogradGraph(__demo.ctx.params(), "physics").totalNodes'), f'{tag}: drawn graph nodes == model.autogradGraph ({n0})')
            K('sageBlend', 0); K('pdeOrder', 3); K('nResiduals', 8); pg.wait_for_timeout(700); shot('13d-order3-terms8')
            n1 = ev(f'{api}.overhead.before.stats.nodes'); check(n1 > n0, f'{tag}: bigger physics -> bigger BEFORE graph ({n0} -> {n1})'); K('pdeOrder', 2); K('nResiduals', 4)
            K('depth', 8); K('width', 256); K('batch', 16384); pg.wait_for_timeout(700); shot('13e-bigger-net'); K('depth', 4); K('width', 128); K('batch', 4096)
            K('sageBlend', 0.5); pg.wait_for_timeout(500); e5 = ev(EXTRA); K('batch', 65536); pg.wait_for_timeout(600); check(ev(EXTRA) > e5, f'{tag}: AD-memory block follows batch ({e5} -> {ev(EXTRA)})'); K('batch', 4096); K('sageBlend', 0)
            # replay: auto-play collapses by itself
            ev("document.getElementById('sage-overhead-controls').querySelector('button').click()"); pg.wait_for_timeout(7500)
            check(ev('__demo.ctx.knobs.get("sageBlend")') > 0.98, f'{tag}: Replay auto-plays the collapse to 1')
            # ---------------- stage 14
            go('sage-jax', 3500); check(ev('__demo.ctx.state.reveal')['sage.jax'] > 0.98, f'{tag}: stage 14 reveal')
            for k in ('sageCompile', 'sageHalf', 'sageSinglePass'): K(k, False)
            pg.wait_for_timeout(900); shot('14a-all-off'); j0 = ev(f'({{s: {api}.jax.series, st: {api}.jax.state, m: {api}.jax.markers}})')
            check(j0['s']['ms'] == 12.7 and j0['s']['measured'], f'{tag}: all off = 12.7 ms (measured, Sep deck)')
            K('sageCompile', True); pg.wait_for_timeout(1200); shot('14b-compile'); j1 = ev(f'({{st: {api}.jax.state, c: {api}.jax.compileEase}})')
            check(j1['st']['overhead'] == 0 and j0['st']['overhead'] > 0 and j1['st']['total'] < j0['st']['total'] and j1['c'] > 0.95, f'{tag}: compile removes the launch-overhead slab')
            K('sageHalf', True); pg.wait_for_timeout(1200); shot('14c-compile-half'); j2 = ev(f'({{s: {api}.jax.series, st: {api}.jax.state}})')
            check(j2['s']['ms'] == 9.9 and j2['s']['measured'] and not j2['s']['trafficMeasured'], f'{tag}: half = 9.9 ms measured, mid traffic is model')
            K('sageSinglePass', True); pg.wait_for_timeout(1800); shot('14d-all-on'); j3 = ev(f'({{s: {api}.jax.series, st: {api}.jax.state, m: {api}.jax.markers}})')
            check(j3['s']['ms'] == 7.7 and abs(j3['s']['units'] - 31) < 1e-6 and j3['st']['total'] < j2['st']['total'] < j1['st']['total'], f'{tag}: all on = 7.7 ms, 31 units, model time falls each step')
            check(j3['m'][0] > j0['m'][0] + 0.5, f'{tag}: bandwidth-bound op slides right on the roofline ({j0["m"][0]:.2f} -> {j3["m"][0]:.2f})')
            K('sageCompile', False); K('sageHalf', False); pg.wait_for_timeout(600); j4 = ev(f'{api}.jax.series'); K('sageSinglePass', True); pg.wait_for_timeout(300); j5 = ev(f'{api}.jax.series')
            check(not j5['measured'], f'{tag}: single-pass alone is NOT labelled measured'); K('sageSinglePass', False)
            pg.wait_for_timeout(500); ro = pg.inner_text('#hud-readouts'); check('measured, Sep 2026 deck' in ro or tag == 'mobile', f'{tag}: HUD labels the deck series')
            K('bandwidth', 3000); pg.wait_for_timeout(1500); shot('14e-bandwidth3000'); mb = ev(f'{api}.jax.state.bound'); K('bandwidth', 696); K('precision', 'bf16'); pg.wait_for_timeout(1200); shot('14f-bf16'); K('precision', 'fp32')
            check(mb == 'compute', f'{tag}: high bandwidth flips the op to compute-bound in the live scene')
            # ---------------- stage 15
            go('sage-results', 2800); shot('15a-results'); txt = ev(f'{api}.results.panelText')
            for need in ('50x', '303 ms', '6.1 ms', '25.4 min', '1.6 min', '16x', '9–10 min', '~10x', '~51 h', '~1.3 h', '~40x', 'PROJECTED', 'Waterloo', 'Seok-Bum Ko', 'Sayed Muhsin', '12 min', '1.0 min', 'RTX 5080', 'A40'):
                check(need in txt, f'{tag}: results panel shows "{need}"')
            check(txt.count('90x') == 1 and '90x' in ev("document.getElementById('sgr-90').textContent") and 'corrected to 50x' in txt, f'{tag}: 90x only in the correction footnote')
            hl = ev(f'{api}.results.headline'); check(hl == {'perStep': '50x', 'a40': '16x', 'rtx': '~10x', 'bedrock': '~40x', 'pytorchMs': 303, 'oursMs': 6.1}, f'{tag}: headline == SPEC facts {hl}')
            w66 = ev(f'{api}.results.wholeSpeedup()'); K('setupShare', 0.15); pg.wait_for_timeout(500); w15 = ev(f'{api}.results.wholeSpeedup()'); shot('15b-setupshare15')
            check(w15 > w66 and w66 < 50 and w15 < 50, f'{tag}: Amdahl: whole run < per step, falls with setup share ({w66:.1f}x at 66%, {w15:.1f}x at 15%)')
            check('PROJECTED' in ev("document.querySelector('[data-row=bed]').textContent") and ev("document.querySelector('[data-row=bed]').style.display") != 'none', f'{tag}: bedrock row badged PROJECTED')
            K('showProjected', False); pg.wait_for_timeout(400); check(ev("document.querySelector('[data-row=bed]').style.display") == 'none', f'{tag}: showProjected off hides bedrock'); shot('15c-no-projected'); K('showProjected', True)
            K('setupShare', 0.66)
            # ---------------- leak test: rapid knob changes across 12-15
            base = None
            for rnd in range(3):
                for i in range(24):
                    for st in ('sage-overhead', 'sage-jax'):
                        pass
                    K('sageBlend', (i % 5) / 4); K('pdeOrder', 1 + i % 3); K('nResiduals', 1 + i % 8); K('depth', 1 + i % 12); K('batch', [64, 4096, 65536][i % 3]); K('width', [16, 128, 512][i % 3])
                    K('sageCompile', i % 2 == 0); K('sageHalf', i % 3 == 0); K('sageSinglePass', i % 4 == 0); K('bandwidth', [200, 696, 2500][i % 3]); K('setupShare', (i % 10) / 10); pg.wait_for_timeout(40)
                for k, v in (('pdeOrder', 2), ('nResiduals', 4), ('depth', 4), ('batch', 4096), ('width', 128), ('bandwidth', 696), ('sageBlend', 0), ('sageCompile', False), ('sageHalf', False), ('sageSinglePass', False), ('setupShare', 0.66)): K(k, v)
                for st in ('sage-overhead', 'sage-idea', 'sage-jax', 'sage-results', 'sage-overhead'): go(st, 700)
                for i in range(12): K('pdeOrder', 1 + i % 3); K('depth', 1 + i % 10); pg.wait_for_timeout(40)
                K('pdeOrder', 2); K('depth', 4); pg.wait_for_timeout(900)
                if rnd == 1: base = ev(INFO)
            after = ev(INFO); print('renderer.info', base, '->', after)
            # a real leak grows by >= 1 geometry per knob change (dozens per round); tiny slack covers lazily created label textures
            check(after['g'] <= base['g'] + 2 and after['t'] <= base['t'] + 4, f'{tag}: no geometry/texture growth after warm-up {base} -> {after}')
            check(ev("document.querySelectorAll('#sage-console, #sage-results, #sage-jax-chips, #sage-idea-controls, #sage-overhead-controls').length") == 5 and ev("document.querySelectorAll('#sage-style').length") == 1, f'{tag}: DOM nodes do not accumulate')
            check(after['calls'] < 200, f'{tag}: draw calls {after["calls"]} < 200')
            # ---------------- reverse gating
            go('sage-jax', 3200); r = ev('__demo.ctx.state.reveal'); check(r['sage.results'] < 0.02 and r['sage.jax'] > 0.98, f'{tag}: reverse to 14 hides results'); check(ev("document.getElementById('sage-results').style.display") == 'none', f'{tag}: results panel hidden')
            go('sage-overhead', 3200); check(ev("document.getElementById('sage-jax-chips').style.display") == 'none' and ev('__demo.ctx.state.reveal')['sage.jax'] < 0.02, f'{tag}: reverse to 13 hides stage 14')
            K('sageBlend', 1); pg.wait_for_timeout(1800); check(ev(EXTRA) == 0, f'{tag}: blend 1 -> extra block 0')
            go('sage-jax', 3000); check(ev(EXTRA) == 0 and ev('__demo.ctx.modules.memory.group.children.length') > 0, f'{tag}: after SAGE the AD-graph block stays gone')
            go('sage-idea', 3200); full = ev("async () => { const m = await import('/js/scenes/physics_math.js'); return m.costRatios(__demo.ctx.params()).graphBytes; }"); check(abs(ev(EXTRA) - full) / max(1, full) < 1e-6 and full > 0, f'{tag}: back at stage 12 the AD-graph block is full again ({ev(EXTRA)})')
            check(ev("document.getElementById('sage-console').style.display") == 'block', f'{tag}: console back at 12')
            go('autograd-cost', 3200); r = ev('__demo.ctx.state.reveal'); check(r['sage.split'] < 0.02 and r['sage.tape'] < 0.02, f'{tag}: reverse to 11 hides stage 12')
            check(ev("document.getElementById('sage-console').style.display") == 'none' and not ev('__demo.ctx.modules.sage.group.children[0].visible') and not ev('__demo.ctx.modules.sage.group.children[1].visible'), f'{tag}: sage 3D + DOM hidden at stage 11')
            check(abs(ev(EXTRA) - full) / max(1, full) < 1e-6, f'{tag}: stage 11 shows the full autograd block')
            go('bounds', 3000); check(not ev('__demo.ctx.modules.sage.group.children[2].visible'), f'{tag}: sage hidden at 7')
            pg.close()
        b.close()
finally:
    srv.terminate()
check(not errors, f'zero console errors {errors[:3]}')
print('SAGE SMOKE', 'FAIL' if fails else 'PASS'); sys.exit(1 if fails else 0)
