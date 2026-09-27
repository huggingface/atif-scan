#!/usr/bin/env bash
# Answer atif-scan follow-up questions with fast-agent.
#
#   atif-scan JOB --plugin atif_scan.packs.tb21:checks --questions review/
#   tools/ask-fast-agent.sh --model sonnet --questions review/ [--question lookup_used] [--jobs 4]
#   atif-scan JOB --plugin atif_scan.packs.tb21:checks --answers review/ --brief
#
# Each prompt (<input>/<question>.md) is sent once with no shell or subagents
# (--no-shell, --no-subagents) and the question's JSON Schema for structured output.
# With --inspect-tool the model also gets three read-only tools over that one trace
# (trace_outline, read_steps, search_trace; tools/atif_inspect_mcp.py) to look up steps
# the prompt's excerpts don't show. The tools mask secrets and run nothing. The reply goes to
# <input>/<question>.answer.json. Existing answers are kept unless --force. atif-scan
# validates replies when it reads them (--answers); invalid ones are reported, not used.
#
# Prompts contain masked trace text and are sent to the model's provider: only run this
# with a provider you're allowed to send the traces to. Keep the directory out of Git.
# Failures are logged (first error line only) to DIR/ask-errors.log. Each answering
# run's own ATIF trajectory is kept as <input>/<question>.review.atif.json.
set -euo pipefail

usage() {
  sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
  echo
  echo "options: --model MODEL (required)  --questions DIR (default: review)"
  echo "         --question ID (repeatable)  --jobs N (default 4)  --timeout SEC (default 300)"
  echo "         --force (re-ask answered)  --dry-run (list what would be asked)"
  echo "         --inspect-tool (let the model read more of the trace through a read-only"
  echo "           MCP server bound to that one trajectory; needs uv, fetches mcp<2)"
  echo "         --fast-agent CMD (default: fast-agent)"
}

answer_one() {
  local meta="$1" prompt="${1%.json}.md" answer="${1%.json}.answer.json"
  local qid; qid="$(basename "${1%.json}")"
  local schema="$ASK_DIR/schemas/$qid.json" tmp="${answer}.tmp.$$"
  local extra=()
  if [[ "$ASK_INSPECT" == 1 ]]; then
    # Read-only MCP server bound to this one trajectory (no shell, no paths from the model).
    local trace
    trace="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("trace_path",""))' "$meta")"
    if [[ -z "$trace" || ! -f "$trace" ]]; then
      printf '%s\t%s\n' "$meta" "no local trajectory for --inspect-tool" >>"$ASK_DIR/ask-errors.log"
      echo "failed    $meta"; return
    fi
    extra=(--stdio "uv run --project $(printf %q "$ASK_REPO") --with mcp>=1.2,<2 python $(printf %q "$ASK_REPO/tools/atif_inspect_mcp.py") $(printf %q "$trace")")
  fi
  # The answering run's own ATIF trajectory (which tools it called, what it read).
  extra+=(--trajectory-output "${1%.json}.review.atif.json")
  if "$ASK_FA" go --model "$ASK_MODEL" --no-shell --no-subagents --quiet "${extra[@]}" \
      --timeout "$ASK_TIMEOUT" --prompt-file "$prompt" --json-schema "$schema" \
      >"$tmp" 2>"$tmp.err" && [[ -s "$tmp" ]] && ! grep -q '^Error:' "$tmp"; then
    # fast-agent can print a tool-status line before the JSON even with --quiet: keep the
    # last line that is a JSON object (atif-scan --answers validates it either way).
    if grep -q '^{' "$tmp"; then grep '^{' "$tmp" | tail -1 >"$answer"; rm -f "$tmp"; else mv "$tmp" "$answer"; fi
    rm -f "$tmp.err"; echo "answered  $meta"
  else
    # Keep only the first line of fast-agent's error (it may echo prompt text otherwise).
    local why
    why="$(cat "$tmp" "$tmp.err" 2>/dev/null | { grep -m1 -i 'error' || true; } | cut -c1-300)"
    [[ -n "$why" ]] || why="$(cat "$tmp" "$tmp.err" 2>/dev/null | grep -m1 . | cut -c1-300 || true)"
    printf '%s\t%s\n' "$meta" "${why:-no output}" >>"$ASK_DIR/ask-errors.log"  # one write
    rm -f "$tmp" "$tmp.err"; echo "failed    $meta"
  fi
}

if [[ "${1-}" == "__one__" ]]; then  # xargs re-entry: MODEL/DIR/... come from the env
  answer_one "$2"; exit 0
fi

ASK_MODEL="${ASK_MODEL:-}" ASK_DIR="${ASK_DIR:-review}" JOBS=4 ASK_TIMEOUT="${ASK_TIMEOUT:-300}" FORCE=0 DRY=0
ASK_FA="${ASK_FA:-${FAST_AGENT:-fast-agent}}"
ASK_INSPECT="${ASK_INSPECT:-0}"
ASK_REPO="$(cd "$(dirname "$0")/.." && pwd)"
QUESTIONS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) ASK_MODEL="$2"; shift 2 ;;
    --questions) ASK_DIR="$2"; shift 2 ;;
    --question) QUESTIONS+=("$2"); shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --timeout) ASK_TIMEOUT="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --inspect-tool) ASK_INSPECT=1; shift ;;
    --fast-agent) ASK_FA="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$ASK_MODEL" ]] || { echo "--model is required" >&2; usage >&2; exit 2; }
[[ -f "$ASK_DIR/index.jsonl" ]] || { echo "no questions in $ASK_DIR (run atif-scan --questions $ASK_DIR)" >&2; exit 2; }
command -v "$ASK_FA" >/dev/null || { echo "fast-agent not found (--fast-agent CMD)" >&2; exit 2; }

todo=()
for meta in "$ASK_DIR"/*/*.json; do
  # Only question metadata: not answers, the answering runs' own trajectories or temp files.
  [[ "$meta" == *.answer.json || "$meta" == *.review.atif.json || "$meta" == *.tmp.* || "$meta" == "$ASK_DIR"/schemas/* ]] && continue
  qid="$(basename "${meta%.json}")"
  if [[ ${#QUESTIONS[@]} -gt 0 ]] && [[ ! " ${QUESTIONS[*]} " == *" $qid "* ]]; then continue; fi
  [[ $FORCE -eq 0 && -s "${meta%.json}.answer.json" ]] && continue
  todo+=("$meta")
done

echo "${#todo[@]} question(s) to ask with model '$ASK_MODEL' (jobs $JOBS$([[ $ASK_INSPECT == 1 ]] && echo ', read-only trace tool'))" >&2
if [[ $DRY -eq 1 ]]; then printf '%s\n' "${todo[@]}"; exit 0; fi
[[ ${#todo[@]} -gt 0 ]] || exit 0

export ASK_MODEL ASK_DIR ASK_TIMEOUT ASK_FA ASK_INSPECT ASK_REPO
printf '%s\0' "${todo[@]}" | xargs -0 -n 1 -P "$JOBS" "$0" __one__ | tee /dev/stderr \
  | awk '{n[$1]++} END {printf "answered %d · failed %d\n", n["answered"], n["failed"]}' >&2
echo "next: atif-scan <same inputs> --answers $ASK_DIR --brief" >&2
