#!/bin/sh
# Smoke test of the INSTALLED wrappers (not the unit tests): the way Claude Code, Codex and Cursor really start pm1.
# c33243d broke every Claude Code hook for 14 hours because the unit tests did not run cc_hook.py the way pm1-cc-hook does,
# and the hooks swallow their own errors (exit 0), so this checks real output, not just the exit status.
# Usage: deploy/smoke.sh   (scratch database and toolout dir, PM1_JUDGE=off; never touches the live data; exit 1 on any failure)
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN="${PM1_BIN:-$HOME/.local/bin}"
PY="${PM1_PYTHON:-python3}"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
export PM1_DB="$T/smoke.sqlite" PM1_JUDGE=off PM1_TOOLOUT_DIR="$T/toolout"
fail=0
ok()  { echo "ok   $1"; }
bad() { echo "FAIL $1"; fail=1; }

# a scratch database holding one active rule for /smoke/proj and one proposal that must stay a proposal
PYTHONPATH="$ROOT/src" "$PY" - <<PYEOF || { echo "FAIL seeding the scratch database"; exit 1; }
import sqlite3
from pm1 import rules
con = sqlite3.connect("$T/smoke.sqlite"); rules.ensure(con)
rules.add(con, "Smoke rule: always answer with the word pineapple.", projects=["/smoke/proj"])
con.execute("INSERT INTO rules(kind, text, projects, triggers, status, proposed_by, created_at) VALUES "
            "('rule', 'Smoke proposal: must stay a proposal.', '[]', '[]', 'proposed', 'smoke', 'now')")
con.commit()
PYEOF

# hooks: realistic payloads; exit 0, no 'pm1 hook:' error text (the hook swallows exceptions), under 5 s, and the
# session-start context really carries the active rule as valid hook JSON
run_hook() {   # name payload
  t0=$(date +%s)
  printf '%s' "$2" | "$BIN/pm1-cc-hook" "$1" >"$T/out" 2>"$T/err"; rc=$?
  t1=$(date +%s)
  [ $rc -eq 0 ] && ! grep -qE "Traceback|pm1 hook:" "$T/err" && [ $((t1 - t0)) -le 5 ] && return 0
  echo "     $1: rc=$rc $(tail -n 2 "$T/err" | tr '\n' ' ')"; return 1
}
run_hook session-start '{"cwd":"/smoke/proj","session_id":"smoke","source":"startup"}' || bad "pm1-cc-hook session-start"
if "$PY" -c "import json,sys; d=json.load(open('$T/out')); assert 'pineapple' in d['hookSpecificOutput']['additionalContext']" 2>/dev/null; then
  ok "pm1-cc-hook session-start (active rule in the context, valid JSON)"
else bad "pm1-cc-hook session-start: no valid hook JSON carrying the active rule"; fi
run_hook pre-tool-use '{"cwd":"/smoke/proj","session_id":"smoke","tool_name":"Edit","tool_input":{"file_path":"/smoke/proj/x.py","new_string":"x"}}' \
  && ok "pm1-cc-hook pre-tool-use" || bad "pm1-cc-hook pre-tool-use"
run_hook post-tool-use '{"cwd":"/smoke/proj","session_id":"smoke","tool_name":"Bash","tool_input":{"command":"ls"},"tool_response":{"stdout":"a\nb","stderr":""}}' \
  && ok "pm1-cc-hook post-tool-use" || bad "pm1-cc-hook post-tool-use"
run_hook post-tool-failure '{"cwd":"/smoke/proj","session_id":"smoke","tool_name":"Bash","tool_input":{"command":"pytest -q"},"error":"boom"}' \
  && ok "pm1-cc-hook post-tool-failure" || bad "pm1-cc-hook post-tool-failure"

# pm1-rule: list works; approve must refuse (no terminal) AND leave the proposal a proposal
"$BIN/pm1-rule" --db "$T/smoke.sqlite" list >"$T/out" 2>"$T/err" && ok "pm1-rule list" || bad "pm1-rule list"
"$BIN/pm1-rule" --db "$T/smoke.sqlite" approve 2 </dev/null >"$T/out" 2>"$T/err"; rc=$?
st=$("$PY" -c "import sqlite3; print(sqlite3.connect('$T/smoke.sqlite').execute('select status from rules where id=2').fetchone()[0])")
if [ $rc -ne 0 ] && [ "$st" = proposed ] && grep -q "terminal" "$T/err"; then ok "pm1-rule approve refuses without a terminal (proposal stays proposed)"
else bad "pm1-rule approve: rc=$rc status=$st (the human-only gate is not working)"; fi

# MCP server: starts, lists its tools (the installer names it pm1-mcp, the development machine's wrapper pm1-mcp-merge;
# the first start of a fresh install may download the embedding model, hence the generous timeout)
MCP="$BIN/pm1-mcp-merge"; [ -x "$MCP" ] || MCP="$BIN/pm1-mcp"
printf '%s\n%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | timeout 240 "$MCP" >"$T/out" 2>"$T/err"
grep -q memory_recall "$T/out" && ok "$(basename "$MCP") initialize + tools/list" || bad "$(basename "$MCP"): no memory_recall in tools/list"

[ $fail -eq 0 ] && echo "smoke: all ok" || echo "smoke: FAILED"
exit $fail
