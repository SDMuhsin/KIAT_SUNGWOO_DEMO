#!/bin/sh
# SAGE demo launcher (macOS / Linux). Tries python3, python, then npx serve. Opens the default browser.
cd "$(dirname "$0")" || exit 1
for py in python3 python; do
  if command -v "$py" >/dev/null 2>&1 && "$py" -c 'import sys; sys.exit(0 if sys.version_info[0] == 3 else 1)' 2>/dev/null; then
    echo "Starting with $py ..."
    exec "$py" tools/serve.py
  fi
done
if command -v npx >/dev/null 2>&1; then
  echo "Python 3 not found; trying npx serve (needs internet the first time) ..."
  URL="http://localhost:8000/demo/index.html"
  (sleep 2; (xdg-open "$URL" || open "$URL") >/dev/null 2>&1) &
  exec npx --yes serve -l 8000 .
fi
echo "No Python 3 and no Node.js found. Install Python 3, or open demo/sage-demo.html directly (single file, offline)."
exit 1
