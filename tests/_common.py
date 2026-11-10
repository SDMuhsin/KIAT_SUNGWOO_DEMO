"""Shared helpers for new tests (cross-platform: no POSIX path assumptions)."""
import glob, os, pathlib, socket, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
GL_ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist']


def free_port():
    s = socket.socket(); s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]; s.close(); return port


def serve(root=ROOT):
    port = free_port()
    srv = subprocess.Popen([sys.executable, str(ROOT / 'tools' / 'serve.py'), '--port', str(port), '--no-browser'], cwd=str(root),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    return srv, port


def launch(p):
    """Launch chromium; falls back to $CHROMIUM_PATH or any playwright-cache chromium on version mismatch."""
    try:
        return p.chromium.launch(args=GL_ARGS)
    except Exception:
        home = os.path.expanduser('~')
        cands = [os.environ.get('CHROMIUM_PATH')] + sorted(
            glob.glob(os.path.join(home, '.cache', 'ms-playwright', 'chromium_headless_shell-*', '*', 'chrome-headless-shell*'))
            + glob.glob(os.path.join(home, '.cache', 'ms-playwright', 'chromium-*', 'chrome-linux*', 'chrome')))
        return p.chromium.launch(executable_path=next(c for c in cands if c), args=GL_ARGS)
