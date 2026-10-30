#!/bin/sh
# Report findings after an edit. This hook never blocks, so it always exits 0.
client="${1:-claude}"
root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
bin="$root/.venv/bin"
[ -x "$bin/python" ] || exit 0
cd "$root" || exit 0
hook() { "$bin/python" -m model_finder.cli.hook "$@"; }

findings=""
add() { [ -n "$1" ] && findings="$findings$1
"; }
files="$(hook files)"
old_ifs="$IFS"
IFS='
'
for file in $files; do
  [ -f "$file" ] || continue
  case "$file" in
    *.py)
      if [ -x "$bin/ruff" ]; then
        "$bin/ruff" format --force-exclude --quiet "$file" >/dev/null 2>&1
        add "$("$bin/ruff" check --force-exclude --quiet --output-format=concise "$file" 2>&1)"
      fi
      if [ -x "$bin/basedpyright" ]; then
        add "$("$bin/basedpyright" --level error "$file" 2>&1 | grep ' - error')"
      fi
      ;;
    */results.json|results.json)
      add "$file: the engine writes this file. Fix: run \`experiment rebuild-results $(dirname "$file")\`."
      ;;
    *)
      experiment="$(hook experiment "$file")"
      if [ -n "$experiment" ] && ! output="$("$bin/experiment" validate "$experiment" 2>&1)"; then
        add "$output"
      fi
      ;;
  esac
done
IFS="$old_ifs"
printf '%s' "$findings" | hook report "$client"
exit 0
