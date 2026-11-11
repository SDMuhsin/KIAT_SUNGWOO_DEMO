#!/usr/bin/env python3
"""ONE smoke test for the single-scene demo. Run from anywhere:  python tests/smoke.py [--shots DIR]
Needs `pip install playwright` + a chromium (or $CHROMIUM_PATH). Serves the repo over http, ALSO loads the single-file bundle from file://
(rebuilt first). Checks: zero console errors / page errors / external requests, SAGE toggle readouts decrease monotonically and restore,
knobs change counts, scrub works, reduced motion, no geometry/texture growth, no horizontal overflow at 1280x720 and 390x844. Writes screenshots."""
import argparse, json, os, pathlib, subprocess, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import ROOT, serve, launch
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser(); ap.add_argument('--shots', default=str(ROOT / 'llmdocs' / 'shots' / 'demo')); ap.add_argument('--slow', type=float, default=1.0)
args = ap.parse_args(); shots = pathlib.Path(args.shots); shots.mkdir(parents=True, exist_ok=True)
fails = []
def check(cond, msg):
    print(('ok   ' if cond else 'FAIL ') + msg)
    if not cond: fails.append(msg)

subprocess.run([sys.executable, str(ROOT / 'tools' / 'build_single.py')], check=True, stdout=subprocess.DEVNULL)
bundle = ROOT / 'demo' / 'sage-demo.html'
txt = bundle.read_text(encoding='utf-8')
import re
check('type="module"' not in txt and 'importmap' not in txt and not re.search(r'(src|href)=["\']https?:', txt), 'bundle: single classic script, no importmap, no remote src/href')

def st(pg): return pg.evaluate('window.__sage.state()')
def wait_for(pg, js, timeout=120000): pg.wait_for_function(js, timeout=timeout, polling=100)

def open_page(b, url, w, h, tag, **ctx):
    c = b.new_context(viewport={'width': w, 'height': h}, **ctx); pg = c.new_page()
    pg.errors, pg.external = [], []
    pg.on('console', lambda m: pg.errors.append(m.type + ': ' + m.text) if m.type in ('error', 'warning') else None)
    pg.on('pageerror', lambda e: pg.errors.append('pageerror: ' + str(e)))
    ok = ('http://127.0.0.1', 'file://', 'data:', 'blob:', 'about:')
    pg.on('request', lambda r: pg.external.append(r.url) if not r.url.startswith(ok) else None)
    pg.goto(url); wait_for(pg, 'window.__sage && window.__sage.state().nodes > 0', 60000); pg.wait_for_timeout(800)
    return pg

def toggle_sequence(pg, tag):
    """Click SAGE, sample the readouts until the transition ends; assert monotone down; click again; assert monotone up and exact restore."""
    pg.evaluate('window.__sage.setU(0.55)')
    s0 = st(pg); check(s0['nodes'] == 648 and s0['matmuls'] == 24, f'{tag}: baseline readouts nodes=648 matmuls=24 (got {s0["nodes"]}, {s0["matmuls"]})')
    keys = ['nodes', 'matmuls', 'memory', 'stepTime']
    def run(target):
        pg.click('#sage-btn'); seq = [st(pg)]; n = 0
        while n < 2000:
            pg.wait_for_timeout(60); seq.append(st(pg)); n += 1
            if abs(seq[-1]['s'] - target) < 1e-6 and n > 3: break
        return seq
    down = run(1)
    for k in keys:
        v = [x[k] for x in down]; check(all(a >= b for a, b in zip(v, v[1:])) and v[-1] < v[0], f'{tag}: {k} decreases monotonically ({v[0]:.4g} -> {v[-1]:.4g}, {len(v)} samples)')
    check(len({x['nodes'] for x in down}) > 4, f'{tag}: transition is gradual (>4 distinct node counts)')
    f = down[-1]; check(f['nodes'] == 30 and f['matmuls'] == 10, f'{tag}: SAGE final nodes=30 matmuls=10 (got {f["nodes"]}, {f["matmuls"]})')
    up = run(0)
    for k in keys:
        v = [x[k] for x in up]; check(all(a <= b for a, b in zip(v, v[1:])) and v[-1] > v[0], f'{tag}: {k} increases monotonically on toggle back')
    r = up[-1]; check(all(abs(r[k] - s0[k]) <= 1e-9 * max(1, abs(s0[k])) for k in keys), f'{tag}: toggling back restores the baseline exactly ' + ('' if all(r[k] == s0[k] for k in keys) else str({k: (s0[k], r[k]) for k in keys})))

with sync_playwright() as p:
    b = launch(p)
    srv, port = serve()
    try:
        # ---------- desktop over http: functional ----------
        pg = open_page(b, f'http://127.0.0.1:{port}/demo/index.html', 1280, 720, 'http')
        toggle_sequence(pg, 'http')
        # knobs change counts (through the real slider UI for batch, the API for the rest)
        c0 = st(pg)['counts']
        pg.evaluate("() => { const i = document.querySelector('#knob-batch'); i.value = 1000; i.dispatchEvent(new Event('input', {bubbles:true})); }"); pg.wait_for_timeout(400)
        c1 = st(pg)['counts']; check(c1['data'] > c0['data'], f'knob batch (UI slider) raises drawn data points {c0["data"]} -> {c1["data"]}')
        pg.evaluate("window.__sage.knobs.set('resPoints', 65536)"); pg.wait_for_timeout(300)
        c2 = st(pg)['counts']; check(c2['res'] > c1['res'], f'knob resPoints raises residual points {c1["res"]} -> {c2["res"]}')
        pg.evaluate("window.__sage.setS(0)")
        base = st(pg); n_base = base['nodes']
        pg.evaluate("window.__sage.knobs.set('pdeOrder', 3)"); pg.wait_for_timeout(300); check(st(pg)['nodes'] > n_base, 'knob derivative order 3 grows the physics graph')
        pg.evaluate("window.__sage.knobs.set('pdeOrder', 1)"); pg.wait_for_timeout(300); check(st(pg)['nodes'] < n_base, 'knob derivative order 1 shrinks the physics graph')
        pg.evaluate("window.__sage.knobs.set('pdeOrder', 2)")
        pg.evaluate("window.__sage.knobs.set('depth', 8)"); pg.wait_for_timeout(300); d8 = st(pg); check(d8['nodes'] > n_base and d8['matmuls'] > 24, f'knob depth 8: nodes {d8["nodes"]}, matmuls {d8["matmuls"]}')
        pg.evaluate("window.__sage.knobs.set('depth', 4)")
        m0 = st(pg)['memory']
        pg.evaluate("window.__sage.knobs.set('precision', 'bf16')"); pg.wait_for_timeout(300); check(st(pg)['memory'] < m0, 'knob precision bf16 lowers memory')
        pg.evaluate("window.__sage.knobs.set('precision', 'fp32')")
        pg.evaluate("window.__sage.knobs.set('adam', false)"); pg.wait_for_timeout(300); check(st(pg)['memory'] < m0, 'knob Adam off lowers memory')
        pg.evaluate("window.__sage.knobs.set('adam', true)")
        t0 = st(pg)['stepTime']; pg.evaluate("window.__sage.knobs.set('bandwidth', 100)"); pg.wait_for_timeout(300); check(st(pg)['stepTime'] > t0, 'knob bandwidth down raises step time')
        pg.evaluate("window.__sage.knobs.set('bandwidth', 696)")
        pg.evaluate("window.__sage.knobs.set('width', 256)"); pg.wait_for_timeout(300); check(st(pg)['memory'] > m0, 'knob width raises memory')
        pg.evaluate("window.__sage.knobs.set('width', 128)"); pg.evaluate("window.__sage.knobs.set('batch', 4096)"); pg.evaluate("window.__sage.knobs.set('resPoints', 4096)"); pg.wait_for_timeout(300)
        # scrub
        pg.evaluate("() => { const i = document.querySelector('#scrub'); i.value = 120; i.dispatchEvent(new Event('input', {bubbles:true})); }"); pg.wait_for_timeout(300)
        a = st(pg); pg.evaluate("() => { const i = document.querySelector('#scrub'); i.value = 600; i.dispatchEvent(new Event('input', {bubbles:true})); }"); pg.wait_for_timeout(300); c = st(pg)
        check(abs(a['u'] - 0.12) < 0.01 and abs(c['u'] - 0.6) < 0.01 and not c['playing'] and a['phase'] != c['phase'], f'scrub moves the step and pauses ({a["phase"]} -> {c["phase"]})')
        pg.click('#btn-replay'); pg.wait_for_timeout(500); r = st(pg); check(r['playing'] and r['u'] < 0.2, 'Replay restarts and plays')
        pg.click('#btn-play'); check(not st(pg)['playing'], 'Pause pauses')
        pg.evaluate("() => { const i = document.querySelector('#sage-scrub'); i.value = 500; i.dispatchEvent(new Event('input', {bubbles:true})); }"); pg.wait_for_timeout(300)
        mid = st(pg); check(0 < mid['nodes'] - 30 < 618 and abs(mid['s'] - 0.5) < 1e-6, f'SAGE scrub holds a mid-transition state (nodes {mid["nodes"]})')
        pg.evaluate("window.__sage.setS(0)")
        # no GPU resource growth after warm-up
        pg.click('#btn-replay'); pg.wait_for_timeout(1500); m_a = st(pg)['mem']
        for kv in [('batch', 20000), ('depth', 6), ('width', 64), ('pdeOrder', 3), ('depth', 4), ('width', 128), ('pdeOrder', 2), ('batch', 4096)]: pg.evaluate(f"window.__sage.knobs.set('{kv[0]}', {kv[1]})"); pg.wait_for_timeout(150)
        pg.click('#sage-btn'); pg.wait_for_timeout(2500); pg.click('#sage-btn'); pg.wait_for_timeout(2500); pg.click('#btn-replay'); pg.wait_for_timeout(1500)
        m_b = st(pg)['mem']; check(m_a == m_b, f'no geometry/texture growth after warm-up {m_a} -> {m_b}')
        check(pg.evaluate('document.documentElement.scrollWidth <= innerWidth && document.documentElement.scrollHeight <= innerHeight'), '1280x720: no page overflow')
        bt = pg.evaluate("(() => { const r = document.querySelector('#sage-btn').getBoundingClientRect(); return r.top >= 0 && r.bottom <= innerHeight && r.right <= innerWidth && r.width >= 150 && r.height >= 40; })()"); check(bt, '1280x720: big SAGE button fully visible')
        check(not pg.errors, 'http desktop: zero console errors/warnings ' + str(pg.errors[:3])); check(not pg.external, 'http desktop: no external requests ' + str(pg.external[:3]))
        # screenshots (desktop)
        def shot(pg, name, u, s):
            pg.evaluate(f'window.__sage.setS({s}); window.__sage.setU({u})'); pg.wait_for_timeout(1200); pg.screenshot(path=str(shots / name))
        for tag, page in [('desktop', pg)]:
            shot(page, f'{tag}_baseline.png', 0.0, 0); shot(page, f'{tag}_mid_forward.png', 0.2, 0); shot(page, f'{tag}_mid_backward.png', 0.5, 0)
            shot(page, f'{tag}_sage_mid.png', 0.5, 0.5); shot(page, f'{tag}_sage_final.png', 0.5, 1.0)
        pg.context.close()

        # ---------- mobile over http ----------
        pg = open_page(b, f'http://127.0.0.1:{port}/demo/index.html', 390, 844, 'mobile', is_mobile=True, has_touch=True, device_scale_factor=2)
        check(pg.evaluate('document.documentElement.scrollWidth <= innerWidth && document.documentElement.scrollHeight <= innerHeight'), '390x844: no page overflow')
        bt = pg.evaluate("(() => { const r = document.querySelector('#sage-btn').getBoundingClientRect(); return r.top >= 0 && r.bottom <= innerHeight && r.right <= innerWidth && r.width >= 120; })()"); check(bt, '390x844: SAGE button fully visible')
        check(pg.evaluate("document.querySelector('#knob-panel').classList.contains('collapsed')"), '390x844: knob panel starts collapsed')
        for name, u, s in [('baseline', 0.0, 0), ('mid_forward', 0.2, 0), ('mid_backward', 0.5, 0), ('sage_mid', 0.5, 0.5), ('sage_final', 0.5, 1.0)]: shot(pg, f'mobile_{name}.png', u, s)
        pg.click('#knob-toggle'); pg.wait_for_timeout(500); pg.screenshot(path=str(shots / 'mobile_knobs_open.png'))
        check(not pg.errors, 'http mobile: zero console errors/warnings ' + str(pg.errors[:3]))
        pg.context.close()

        # ---------- reduced motion ----------
        pg = open_page(b, f'http://127.0.0.1:{port}/demo/index.html', 1280, 720, 'rm', reduced_motion='reduce')
        s = st(pg); check(s['reduced'] and not s['playing'], 'reduced motion: starts paused')
        pg.click('#sage-btn'); pg.wait_for_timeout(500); s = st(pg); check(s['s'] == 1 and s['nodes'] == 30, 'reduced motion: SAGE applies without animation')
        check(not pg.errors, 'reduced motion: zero console errors'); pg.context.close()

        # ---------- single-file bundle over file:// ----------
        pg = open_page(b, bundle.as_uri(), 1280, 720, 'file')
        toggle_sequence(pg, 'file')
        check(not pg.errors, 'file:// bundle: zero console errors/warnings ' + str(pg.errors[:3])); check(not pg.external, 'file:// bundle: no network requests ' + str(pg.external[:3]))
        pg.context.close()
    finally:
        srv.terminate(); b.close()

print('PASS' if not fails else 'FAILED %d: %s' % (len(fails), '; '.join(fails)))
sys.exit(1 if fails else 0)
