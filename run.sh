#!/usr/bin/env bash
# LAMOS installer and runner for macOS and Linux.
# Requirements: git and Python 3.10-3.13 with Tk support
#   macOS: the python.org installer includes Tk; with Homebrew run `brew install python-tk`
#   Ubuntu/Debian: sudo apt install python3-tk python3-venv
set -e
PY=${PYTHON:-python3}
if [ ! -f lamos_app.py ]; then
  if [ -d lamos/.git ]; then
    echo "[1/4] Updating ./lamos ..."; git -C lamos pull
  else
    echo "[1/4] Cloning https://github.com/manjukcus/lamos ..."; git clone https://github.com/manjukcus/lamos.git
  fi
  cd lamos
fi
"$PY" -c "import tkinter" 2>/dev/null || { echo "ERROR: $PY has no Tk support (see the comments at the top of run.sh)."; exit 1; }
[ -d venv ] || { echo "[2/4] Creating the virtual environment ..."; "$PY" -m venv venv; }
# shellcheck disable=SC1091
source venv/bin/activate
echo "[3/4] Installing the required packages ..."
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt
echo "[4/4] Starting LAMOS ..."
python lamos_app.py
