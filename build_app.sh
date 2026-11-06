#!/usr/bin/env bash
# Build Jarvis.app — a double-clickable macOS application.
#
# Produces dist/Jarvis.app. Drag it to /Applications and it behaves like any
# other Mac app: Dock icon, no terminal, no server to start.
#
# The bundle is unsigned, so the first launch needs a right-click → Open (or
# System Settings → Privacy & Security → Open Anyway). Gatekeeper blocks a
# plain double-click on unsigned apps; this is expected, not a failure.

set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
APP_NAME="Jarvis"

if [[ ! -d "$VENV" ]]; then
    echo "No virtualenv found. Creating one…"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet -r requirements.txt
fi

echo "Checking build dependencies…"
"$VENV/bin/pip" install --quiet "flet[all]>=0.24" watchdog pyinstaller

echo "Running tests before packaging…"
if ! "$VENV/bin/python" -m pytest -q; then
    echo
    echo "Tests failed. Packaging anyway would ship a known-broken app."
    exit 1
fi

echo "Building ${APP_NAME}.app…"
rm -rf build "dist/${APP_NAME}.app"

# The sentence-transformers backend is optional. If it is installed, it has to
# be bundled explicitly -- PyInstaller cannot see the model/config files torch
# and transformers load by path at runtime, so a bundle built without this
# starts fine and then fails on the first index, which looks like a hang.
ST_ARGS=""
if "$VENV/bin/python" -c "import sentence_transformers" 2>/dev/null; then
    echo "  …including sentence-transformers (large; adds several minutes)"
    ST_ARGS="--collect-all sentence_transformers --collect-all transformers"
    ST_ARGS="$ST_ARGS --collect-all tokenizers --collect-all safetensors"
    ST_ARGS="$ST_ARGS --collect-all torch --collect-all sklearn --collect-all scipy"
fi

# --windowed produces a .app rather than a console binary.
# app/ and web/ are added as data because PyInstaller's import analysis does
# not see files loaded at runtime by path.
"$VENV/bin/pyinstaller" \
    --name "$APP_NAME" \
    --windowed \
    --noconfirm \
    --clean \
    --osx-bundle-identifier "com.beascochea.jarvis" \
    --add-data "app:app" \
    --add-data "web:web" \
    --hidden-import "anthropic" \
    --hidden-import "pypdf" \
    --hidden-import "watchdog.observers.fsevents" \
    --collect-all "flet" \
    --collect-all "flet_desktop" \
    ${ST_ARGS:-} \
    jarvis.py

if [[ -d "dist/${APP_NAME}.app" ]]; then
    echo
    echo "Built dist/${APP_NAME}.app"
    echo
    echo "Install:  mv dist/${APP_NAME}.app /Applications/"
    echo "First run: right-click the app → Open (unsigned bundle)."
else
    echo "Build did not produce dist/${APP_NAME}.app" >&2
    exit 1
fi
