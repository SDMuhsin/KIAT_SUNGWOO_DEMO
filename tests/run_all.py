#!/usr/bin/env python3
"""Run every test: node unit tests (tests/*.test.mjs) then every tests/*.smoke.py. Cross-platform:  python tests/run_all.py [--skip-smoke]
Needs `node` on PATH (or set NODE=path) and, for smoke tests, `pip install playwright` + `playwright install chromium`."""
import glob, os, shutil, subprocess, sys, pathlib
here = pathlib.Path(__file__).resolve().parent
skip = '--skip-smoke' in sys.argv
node = os.environ.get('NODE') or shutil.which('node')
bad = []
if node:
    for f in sorted(glob.glob(str(here / '*.test.mjs'))):
        r = subprocess.run([node, f]); print(('PASS ' if r.returncode == 0 else 'FAIL ') + os.path.basename(f)); r.returncode and bad.append(f)
else:
    print('SKIP node tests: node not found (install Node.js or set NODE)'); bad.append('node missing')
if not skip:
    for f in sorted(glob.glob(str(here / '*.smoke.py'))) + [str(here / 'smoke.py')]:
        r = subprocess.run([sys.executable, f]); print(('PASS ' if r.returncode == 0 else 'FAIL ') + os.path.basename(f)); r.returncode and bad.append(f)
print('ALL PASS' if not bad else 'FAILED: ' + ', '.join(os.path.basename(b) for b in bad)); sys.exit(1 if bad else 0)
