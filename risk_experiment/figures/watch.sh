#!/usr/bin/env bash
# Live-render figure PDFs on save.
#
#   ./risk_experiment/figures/watch.sh
#
# Then just edit any risk_experiment/figures/figure_*.py and save -- the matching
# PDF ingredient(s) re-render automatically into revision/figures/, so a PDF
# viewer (Preview / Affinity / the IDE) shows the fresh version on refresh. No
# manual `python -m ...` step, and nobody has to rasterize + eyeball every tweak.
#
# Editing the shared modules (style.py / verify.py) re-renders ALL figures, since
# they affect every panel. Render errors are printed inline (last 15 log lines).
#
# Requires: watchexec (`brew install watchexec`) and the risk7t conda env.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$HOME/mambaforge/envs/risk7t/bin/python"
FIGDIR="$REPO/risk_experiment/figures"
LOG=/tmp/figwatch.log

render_module () {  # $1 = module stem, e.g. figure_03_alt_models
  printf '↻ %s … ' "$1"
  if ( cd "$REPO" && "$PY" -m "risk_experiment.figures.$1" ) >"$LOG" 2>&1; then
    echo "✓"
  else
    echo "✗"; tail -n 15 "$LOG"
  fi
}

render_all () {
  for f in "$FIGDIR"/figure_*.py; do render_module "$(basename "$f" .py)"; done
}

# Render whatever was just saved: the most-recently-modified .py in FIGDIR is the
# file that triggered us. style.py / verify.py are shared -> render everything.
on_change () {
  local newest stem
  newest="$(ls -t "$FIGDIR"/*.py 2>/dev/null | head -1)"
  [ -z "$newest" ] && return
  stem="$(basename "$newest" .py)"
  case "$stem" in
    style|verify) render_all ;;
    figure_*)     render_module "$stem" ;;
    *)            : ;;  # ignore __init__.py etc.
  esac
}

# watchexec re-invokes this script with --once on each change (the function
# definitions above are re-evaluated each time, so this stays self-contained).
if [ "${1:-}" = "--once" ]; then on_change; exit 0; fi

command -v watchexec >/dev/null 2>&1 || {
  echo "watchexec not found -- install it:  brew install watchexec" >&2; exit 1; }

echo "👀 Watching $FIGDIR"
echo "   Edit a figure_*.py and save; its PDF refreshes in revision/figures/."
echo "   (style.py / verify.py -> re-render all.)  Ctrl-C to stop."
exec watchexec --quiet --postpone --watch "$FIGDIR" --exts py --debounce 300ms \
  -- "$0" --once
