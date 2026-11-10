#!/usr/bin/env python3
"""Tiny static server for the SAGE demo (stdlib only; Windows/macOS/Linux, Python 3.7+).
Why not plain `python -m http.server`? On some Windows machines the registry maps .js to text/plain, which makes the browser
refuse ES modules. This wraps http.server and forces correct MIME types, picks a free port (8000 first) and opens the browser.
Usage: python tools/serve.py [--port 8000] [--no-browser]"""
import argparse, http.server, os, socketserver, sys, threading, webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = dict(http.server.SimpleHTTPRequestHandler.extensions_map)
    extensions_map.update({'.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.html': 'text/html', '.json': 'application/json', '.svg': 'image/svg+xml'})

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, default=8000); ap.add_argument('--no-browser', action='store_true')
    a = ap.parse_args()
    os.chdir(ROOT)
    srv = None
    for port in [a.port] + list(range(a.port + 1, a.port + 100)) + [0]:
        try:
            srv = Server(('127.0.0.1', port), Handler); break
        except OSError:
            continue
    if srv is None:
        print('Could not bind any port.'); return 2
    url = 'http://127.0.0.1:%d/index.html' % srv.server_address[1]
    print('SAGE demo running at ' + url + '\nPress Ctrl+C to stop.')
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
