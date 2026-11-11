#!/usr/bin/env python3
"""Build demo/sage-demo.html: ONE self-contained file (CSS + JS + vendored three.js inlined).

Works by double-click (file://) with no server. Stdlib only; runs on Python 3.7+ (Windows/mac/linux).

How: the ES modules (js/**, vendor/three) are transformed into a tiny CommonJS-like registry inside a single
classic <script> (an IIFE). `import`/`export` statements are rewritten, dynamic `import()` is routed through
the registry, and the entry module runs inside an async function (it uses top-level await). Exports are live
getters, so circular imports behave like real ES modules. No importmap, no blob/data URLs, no CORS.

Usage:  python tools/build_single.py [-o demo/sage-demo.html]
"""
import argparse, os, re, sys, posixpath

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = 'demo/js/main.js'
BARE = {'three': 'vendor/three/build/three.module.js'}
BARE_PREFIX = {'three/addons/': 'vendor/three/examples/jsm/'}

IMPORT_RE = re.compile(r'^[ \t]*import\s+(?:([\w$]+)\s*,?\s*)?(?:\*\s+as\s+([\w$]+)|\{([^}]*)\})?\s*from\s*[\'"]([^\'"]+)[\'"]\s*;?[ \t]*$', re.M)
SIDE_IMPORT_RE = re.compile(r'^[ \t]*import\s*[\'"]([^\'"]+)[\'"]\s*;?[ \t]*$', re.M)
EXPORT_DECL_RE = re.compile(r'^export\s+(?:async\s+)?(function\*?|class|const|let|var)\s+([\w$]+)', re.M)
EXPORT_LIST_RE = re.compile(r'^export\s*\{([^}]*)\}\s*;?[ \t]*$', re.M)
EXPORT_DEFAULT_RE = re.compile(r'^export\s+default\s+(?:(async\s+function\*?|function\*?|class)\s+([\w$]+)|)', re.M)
DYN_IMPORT_RE = re.compile(r'(?<![\w$.])import\s*\(')


def resolve(spec, importer):
    if spec in BARE:
        return BARE[spec]
    for pre, tgt in BARE_PREFIX.items():
        if spec.startswith(pre):
            return tgt + spec[len(pre):]
    if spec.startswith('.'):
        return posixpath.normpath(posixpath.join(posixpath.dirname(importer), spec))
    raise SystemExit('cannot resolve %r from %s' % (spec, importer))


def read(rel):
    with open(os.path.join(ROOT, *rel.split('/')), encoding='utf-8') as f:
        return f.read().replace('\r\n', '\n')


def names(lst):
    out = []
    for part in lst.split(','):
        part = part.strip()
        if not part:
            continue
        m = re.match(r'([\w$]+)(?:\s+as\s+([\w$]+))?$', part)
        if not m:
            raise SystemExit('bad specifier ' + part)
        out.append((m.group(1), m.group(2) or m.group(1)))
    return out


def transform(rel, src):
    deps, exports = [], []   # exports: (exportName, localExpr)
    pre = []

    def imp(m):
        dflt, ns, lst, spec = m.groups()
        tgt = resolve(spec, rel)
        deps.append(tgt)
        v = '__m%d' % len(deps)
        bits = ['const %s = __r(%r);' % (v, tgt)]
        if ns:
            bits.append('const %s = %s;' % (ns, v))
        if dflt:
            bits.append('const %s = %s.default;' % (dflt, v))
        if lst is not None:
            for a, b in names(lst):
                bits.append('const %s = %s.%s;' % (b, v, a))
        return ' '.join(bits)

    def side(m):
        tgt = resolve(m.group(1), rel)
        deps.append(tgt)
        return '__r(%r);' % tgt

    src = IMPORT_RE.sub(imp, src)
    src = SIDE_IMPORT_RE.sub(side, src)

    def edecl(m):
        exports.append((m.group(2), m.group(2)))
        return m.group(0)[len('export '):]
    src = EXPORT_DECL_RE.sub(edecl, src)

    def elist(m):
        for a, b in names(m.group(1)):
            exports.append((b, a))
        return ''
    src = EXPORT_LIST_RE.sub(elist, src)

    def edef(m):
        if m.group(2):
            exports.append(('default', m.group(2)))
            return '%s %s' % (m.group(1), m.group(2)) if m.group(1) else ''
        exports.append(('default', '__default'))
        return 'var __default ='
    src = EXPORT_DEFAULT_RE.sub(edef, src)
    if re.search(r'^\s*export\s', src, re.M):
        raise SystemExit('unhandled export syntax in ' + rel)
    if 'import.meta' in src:
        raise SystemExit('import.meta not supported (' + rel + ')')
    src = DYN_IMPORT_RE.sub('__dyn(', src)
    head = ''.join('Object.defineProperty(__e,%r,{enumerable:true,get:function(){return %s;}});' % (a, b) for a, b in exports)
    is_async = rel == ENTRY
    return deps, head, src, is_async


def collect():
    mods, order, stack = {}, [], [ENTRY]
    while stack:
        rel = stack.pop()
        if rel in mods:
            continue
        deps, head, body, is_async = transform(rel, read(rel))
        mods[rel] = (head, body, is_async)
        order.append(rel)
        stack.extend(reversed(deps))
    return mods, order


def js_str(s):
    return s.replace('</script', '<\\/script').replace('<!--', '<\\!--')


def build(out):
    mods, order = collect()
    parts = []
    parts.append('(function(){"use strict";var __defs={},__cache={};')
    parts.append('function __r(id){var m=__cache[id];if(m)return m.e;var d=__defs[id];if(!d)throw new Error("module not found: "+id);'
                 'var e={};m=__cache[id]={e:e};d(e,id);return e;}')
    parts.append('function __resolve(spec,from){if(spec.charAt(0)!==".")return spec;var p=from.split("/");p.pop();'
                 'spec.split("/").forEach(function(s){if(s===".")return;if(s==="..")p.pop();else p.push(s);});return p.join("/");}')
    for rel in order:
        head, body, is_async = mods[rel]
        fn = 'async function' if is_async else 'function'
        parts.append('\n/* ---- %s ---- */\n__defs[%r]=%s(__e,__id){var __dyn=function(s){try{return Promise.resolve(__r(__resolve(s,__id)));}catch(x){return Promise.reject(x);}};%s\n%s\n};'
                     % (rel, rel, fn, head, js_str(body)))
    parts.append('\nPromise.resolve(__r(%r)).catch(function(e){console.error(e);});' % ENTRY)
    parts.append('})();')
    js = ''.join(parts)

    # the entry module is async: __r returns the exports object immediately while main's top-level await continues.
    html = read('demo/index.html')
    css = read('demo/css/style.css')
    html = re.sub(r'<script type="importmap">.*?</script>\s*', '', html, flags=re.S)
    html = html.replace('<link rel="stylesheet" href="css/style.css">', '<style>\n' + css.replace('</style', '<\\/style') + '\n</style>')
    # classic script placed at end of body (DOM must exist: main.js reads elements at start)
    html = html.replace('<script type="module" src="js/main.js"></script>', '<script>\n' + js + '\n</script>')
    if 'type="module"' in html or 'importmap' in html:
        raise SystemExit('leftover module/importmap in html')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(html)
    return len(order), len(html)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default=os.path.join(ROOT, 'demo', 'sage-demo.html'))
    a = ap.parse_args()
    n, size = build(a.out)
    print('built %s: %d modules, %.2f MB' % (a.out, n, size / 1e6))
