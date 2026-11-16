# pm1 reference manual

> This manual describes what the code does **today**. The design is explained in [DESIGN.md](DESIGN.md) and the
> measurements in [EVALUATION.md](EVALUATION.md) and [RESULTS.md](RESULTS.md).
> Anything not built yet is marked **(planned)**. pm1 is `0.1.0.dev0` and has been run only on one Linux machine.

pm1 keeps a local, searchable record of the work your AI coding agents do: Codex, Claude Code, Cursor, and Cowork cloud
sessions you export. Every agent can then look up what was asked, decided, run and reported in earlier sessions,
including sessions run by *other* agents. It also searches your own documents and keeps the standing rules and
lessons that you approved.

Principles, as implemented:

- **Raw records first.** Captured turns and tool results are stored as they were, after secret redaction. Every index
  (spans, BM25, vectors) is derived from them and can be rebuilt. Answers are original records with pointers
  (`episode_id`, `evidence_id`), never generated summaries. Recall needs no generative model.
- **Records are data, not instructions.** Results come wrapped in a notice saying so. Text an agent writes through
  `save_note` (formerly `memory_capture`) is tagged as agent-written and can never become a standing user instruction.
- **Only you decide rules.** Agents can *propose* a rule or lesson. Only you, in a terminal, can activate, approve,
  edit or revoke one (`pm1-rule`).
- **Local by default.** One SQLite file. No server process except the stdio MCP server your agent starts.
  See [Privacy](#privacy) for the exceptions.

## What you get

**MCP tools** (one stdio server, `pm1-mcp`):

| Tool | What it does |
|---|---|
| `search_sessions` | Search all captured agent sessions (filters: project, host, since/until, roles). Returns raw records with `episode_id`. |
| `open_record` | One record with the turns around it. |
| `session_brief` | Handoff brief of one past session (default: the latest of a project): goal, latest requests, approvals, open tasks, files, commits, last report, each with an `episode_id`. Built from records, no LLM. |
| `code_history` | History of a file or symbol: which sessions edited it, the request behind each edit, failures right after, commits. Callers/callees come from a [graphify](#code-history-and-graphify) graph when one exists. |
| `search_docs`, `cite`, `docs_status` | Search your registered folders (notes, Markdown, PDF, DOCX, XLSX, …) by meaning and keywords. `cite` resolves an `evidence_id` and checks whether the file has changed since it was indexed. |
| `rules` | Your active standing rules and lessons for a project. |
| `propose_rule` | An agent proposes a rule / lesson / recipe, quoting your own words. The quote is checked against a captured turn of yours. It stays inactive until you approve it. |
| `pm1_feedback` | An agent reports whether a result helped. Used only by the usage meter. |
| `save_note`, `search_notes`, `memory_status` | Notes an agent saves on purpose, and a search over them (BM25 + optional dense + optional reranker). Not past sessions: those are `search_sessions`. Renamed 2026-11-15 from `memory_capture` / `memory_recall`; the old names still answer (unlisted), `PM1_NOTE_TOOLS=old` lists them again. |

Results carry a `call_id`. Every call is logged locally (`mcp_calls`) for the usage meter.

**Claude Code hooks** (`pm1.cc_hook`): see [Claude Code hooks](#claude-code-hooks).

**CLIs:** `pm1-mcp`, `pm1-capture`, `pm1-docs`, `pm1-worker`, `pm1-rule`, `pm1-meter` (console scripts from `pyproject.toml`).

## Install

Requirements: Python ≥ 3.10 whose `sqlite3` module has FTS5 (most builds do), and Linux. macOS and Windows have not been
tried; the Cursor capture path is Linux-only today (see [Known limitations](#known-limitations)).

```sh
git clone <repo> precision-memory && cd precision-memory
python3 -m venv .venv
.venv/bin/pip install -e '.[local-embed,docs]'   # local-embed = fastembed (CPU e5); docs = PDF/DOCX/XLSX parsers
```

A PyPI package and a `pm1 install` command that writes config and user units are **(planned)**. Until then, put
`.venv/bin` on your PATH or call the scripts by full path.

pm1 is configured by environment variables ([reference](#configuration-reference)) or by `~/.config/pm1/config.toml`
(same settings, see [Configuration](#configuration-reference)); the environment wins.

**One database by default.** Every tool (`pm1-mcp`, `pm1-worker`, `pm1-capture`, `pm1-docs`, the hooks, `pm1-rule`, `pm1-meter`) opens
`$XDG_STATE_HOME/pm1/memory.sqlite` (`~/.local/state/pm1/memory.sqlite`) unless `PM1_DB` says otherwise. Set `PM1_DB` only to move it:

```sh
export PM1_DB=~/.local/state/pm1/memory.sqlite
```

## CPU-only mode

pm1 needs no GPU. Search quality comes in layers, and each layer is optional:

| Layer | Needs | Set |
|---|---|---|
| BM25 (always on) | nothing | — |
| dense e5-small (multilingual) | `fastembed` (extra `local-embed`), a one-time model download from HuggingFace, CPU | `PM1_DENSE=local` (the default when `fastembed` is installed, else `none`) |
| late-interaction token vectors | an HTTP service (`/v1/encode`, pplx-embed) | `PM1_TOKEN=pplx` (off by default: `PM1_TOKEN=none`) |
| cross-encoder reranker | an HTTP reranker service (llama.cpp `/v1/rerank` or vLLM `/score`) | `PM1_RERANK_*` URLs (unset by default) |

Recommended CPU settings:

```sh
export PM1_DENSE=local          # PM1_TOKEN already defaults to none
export PM1_THREADS=4            # ONNX threads for the local embedder (default 8)
```

Notes on what the code does today:

- Without `fastembed` installed the default is `PM1_DENSE=none`: BM25 only, which works. Install the `local-embed` extra
  to get dense vectors. Setting `PM1_DENSE=local` explicitly without it fails at startup.
- If the dense embedder or a reranker fails at query time, the call still returns. Dense is skipped for 60 s and the
  search falls back to BM25. A failed reranker cools down for 60 s and the fused first-stage order is kept. The
  response says which path was used (`mode`, `route.fallback`).
- Setting `PM1_TOKEN=pplx` without the service queues token-vector jobs that never finish, and the worker stops after
  5 failed batches. Leave `PM1_TOKEN` at its default `none` unless you run that service.
- English-only rerankers are never used on non-English text. The router picks a multilingual one or keeps the fused order.
- The local model file is the int8 AVX512-VNNI ONNX build of `intfloat/multilingual-e5-small`. I have not tested how it behaves on CPUs without VNNI.
- A CPU reranker can be plugged in through `PM1_RERANK_QWEN_CPU` / `PM1_RERANK_BGE_CPU` (any llama.cpp `--rerank`
  server). The latency budget keeps slow rerankers out of the `fast` and `balanced` profiles. A packaged CPU reranker is **(planned)**.

Vectors are computed in the background by the worker. New records are searchable by BM25 right away.

```sh
pm1-worker                       # long-running; or bounded: --until 07:00 --rate 12 --max-load 8 --exit-when-empty
```

## Capturing sessions

```sh
pm1-capture scan                 # one pass
pm1-capture watch -i 60          # keep capturing every 60 s
pm1-capture report --days 7      # what was captured
```

What `pm1-capture` reads (all read-only):

- Codex: `~/.codex/sessions/*/*/*/rollout-*.jsonl`
- Claude Code: `~/.claude/projects/*/*.jsonl`
- Cursor: `~/.config/Cursor/User/globalStorage/state.vscdb`, opened read-only, plus `workspaceStorage/*/workspace.json`
  to map chats to folders. Turn it off with `PM1_CAPTURE_CURSOR=0`.
- The *names* of configured MCP servers, hooks and skills in `~/.codex/config.toml`, `~/.cursor/mcp.json`, `~/.claude.json`
  and `~/.claude/settings.json`, which give a timeline of memory add-ons. Commands and env values are not read.
- Cowork (cloud) sessions only through an explicit import: `pm1-capture import-conversation FILE --session ID [--label L]`
  and `import-usage`.

A client that is not installed is skipped. Sessions are grouped by project (`proj:<working folder>`). Code history
(file touches, commits, failed commands) is extracted during capture unless `PM1_CODE=0`.

### Code history and graphify

`code_history` works from captured tool calls alone. If a repo has a graph built by graphify
(`graphify-out/graph.json`), pm1 also imports its symbols and call edges. Today it looks only in
`~/work/*[/*]/graphify-out/graph.json` and `~/code/*[/*]/graphify-out/graph.json` by default; `PM1_GRAPH_GLOBS` (globs separated by `:`) overrides the list.

## Documents

```sh
pm1-docs add ~/notes             # register a folder; files stay where they are
pm1-docs scan                    # index new/changed files, retire deleted ones
pm1-docs watch --interval 300    # keep scanning
pm1-docs search "what did we decide about X"
pm1-docs cite ev_...
```

PDF, DOCX and XLSX need the `docs` extra. Each passage gets a stable `evidence_id` with a locator (lines / page / sheet+cell).

## Registering the MCP server

Every client starts `pm1-mcp` over stdio. Pass the env vars the server needs, and set `PM1_CLIENT` so the usage meter
can tell clients apart (`claude-code`, `codex`, `cursor`). A small wrapper script that exports your settings and runs
`exec pm1-mcp "$@"` keeps the three registrations identical:

```sh
#!/bin/sh
export PM1_DENSE=local PM1_THREADS=4
export PM1_RERANK_PROFILE=wait PM1_QUALITY_FLOOR=wait   # agents wait up to 8 s for a reranker, if you configured one
exec /path/to/pm1/.venv/bin/pm1-mcp "$@"
```

**Claude Code** (`~/.claude.json`, `mcpServers`; or `claude mcp add`):

```json
"pm1": { "command": "/path/to/pm1-mcp-wrapper", "args": [], "env": { "PM1_CLIENT": "claude-code" } }
```

**Codex** (`~/.codex/config.toml`):

```toml
[mcp_servers.pm1]
command = "/path/to/pm1-mcp-wrapper"
args = []

[mcp_servers.pm1.env]
PM1_CLIENT = "codex"
```

**Cursor** (`~/.cursor/mcp.json`, `mcpServers`):

```json
"pm1": { "command": "/path/to/pm1-mcp-wrapper", "args": [], "env": { "PM1_CLIENT": "cursor" } }
```

To turn off whole tool groups: `PM1_SESSIONS=0` (sessions, rules, propose_rule, feedback, code history), `PM1_CODE=0`, `PM1_DOCS=0`.

## Claude Code hooks

`pm1.cc_hook` adds four hooks. They never block Claude Code, and errors inside a hook are swallowed.

| Event (matcher) | What it does |
|---|---|
| `SessionStart` (`startup\|resume\|clear`) | one line saying pm1's tools exist, plus your standing rules for this project |
| `SessionStart` (`compact`) | catches up on this session's transcript, then injects the handoff brief (raw records with pointers) so a compacted session keeps the exact requests |
| `PostToolUse` (`Bash`) | output over 16 000 chars (`PM1_HOOK_TOOLOUT_LIMIT`): the full output is saved (redacted) under `PM1_TOOLOUT_DIR`, and the model sees head + tail + the file path. It also pushes a lesson once per session when one of its trigger strings appears |
| `PreToolUse` (`Edit\|Write\|MultiEdit\|NotebookEdit`) | if earlier sessions edited this file and then hit an error naming it, says so once; lesson triggers in the edit. Never changes the permission decision |
| `PostToolUseFailure` (`Bash`) | if the same error with the same command was fixed in an earlier session, says what changed in between |

The hook can run as a file (`python src/pm1/cc_hook.py <event>`, what the example wrapper below does, fastest start) or as a
module. A `pm1-cc-hook` console script is **(planned)**. Example wrapper `pm1-cc-hook`:

```sh
#!/bin/sh
export PM1_DB="$HOME/.local/state/pm1/memory.sqlite"
exec /path/to/pm1/.venv/bin/python -m pm1.cc_hook "$@"
```

`~/.claude/settings.json`:

```json
"hooks": {
  "SessionStart": [
    {"matcher": "compact", "hooks": [{"type": "command", "command": "/path/to/pm1-cc-hook session-start", "timeout": 30}]},
    {"matcher": "startup|resume|clear", "hooks": [{"type": "command", "command": "/path/to/pm1-cc-hook session-start", "timeout": 10}]}],
  "PostToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "/path/to/pm1-cc-hook post-tool-use", "timeout": 10}]}],
  "PreToolUse": [{"matcher": "Edit|Write|MultiEdit|NotebookEdit", "hooks": [{"type": "command", "command": "/path/to/pm1-cc-hook pre-tool-use", "timeout": 5}]}],
  "PostToolUseFailure": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "/path/to/pm1-cc-hook post-tool-failure", "timeout": 5}]}]
}
```

Hooks for Codex and Cursor are **(planned)**. Those clients get pm1 through the MCP tools only.

## Standing rules and lessons: `pm1-rule`

**Run the deciding commands (`add`, `approve`, `reject`, `revoke`, `edit`, `renew`, `dismiss`) in your own terminal.** They refuse to run
without a terminal or when an agent environment variable (`CLAUDECODE`, `CODEX_SANDBOX`, ...) is set, so an agent cannot approve its own
proposals, and they do not work from an agent's `!` shell either. This is a speed bump, not a wall: anything that can write the SQLite file
can bypass it.

Three kinds: a **rule** is shown when a session starts in a matching project. A **recipe** is how to run something in
a project, taken from an observed command that passed after the obvious one failed. A **lesson** is a cross-project
pitfall with trigger strings, pushed once per session when a trigger appears in a command, its output or an edit.

```sh
pm1-rule list [--all]                 # active rules / lessons (--all: every status)
pm1-rule pending                      # proposals waiting for you
pm1-rule show ID
pm1-rule approve ID [--text … --project … | --global --trigger … --until 2w --review 4w --why … --do …]
pm1-rule reject ID   |  pm1-rule revoke ID
pm1-rule add "text" [--lesson --trigger STR] [--project P | --global] [--replaces ID] [--when …] [--until …]
pm1-rule edit ID … |  pm1-rule renew ID
pm1-rule conflicts [--all]            # your later messages that may change a standing rule
pm1-rule dismiss CID
pm1-rule suggest [--days 2 --max 5 --dry-run --no-judge]   # propose lessons found in recent records (inactive until approved)
pm1-rule judge-stats                  # how often the judges agreed with your decisions
```

`add`, `approve`, `reject`, `revoke`, `edit`, `renew` and `dismiss` refuse to run when no terminal is attached or when
an agent environment variable is set (`CLAUDECODE`, `CLAUDE_CODE_ENTRYPOINT`, `CODEX_SANDBOX`, `CODEX_THREAD_ID`,
`CURSOR_AGENT`, `GROKBOT_TASK`). This is a speed bump, not a wall: anything that can write the SQLite file can bypass it.

`pm1-rule suggest` needs a reranker service (`PM1_RERANK_QWEN_VLLM` or `PM1_RERANK_QWEN`) or the judges. With
neither, it prints a note and exits 0 (no suggestions that day).

## Privacy

What stays local:

- Everything pm1 stores lives in one SQLite file (`PM1_DB`) plus the tool-output folder of the Claude Code hook.
  Capture opens the clients' files read-only.
- **Redaction:** API keys, tokens and passwords (common key formats, `password=`, bearer headers, …) are cut before
  any record is stored (`Store.add_episode`, document blocks, usage logs). They are cut again from every MCP result,
  from the hook's saved tool output and from anything sent to the judges. The patterns are deliberately simple. They can
  miss a secret in an unusual format. `python -m pm1.redact [--dry-run]` cleans a database written before redaction existed
  (back it up first).
- Agents cannot delete memory through MCP.

What can leave the machine:

1. **Through your agents.** Whatever a pm1 tool returns goes into the agent's context, and that agent's host sends it
   to its model provider. That is the point of the tool, but it means old session records travel with each search.
2. **The embedding model download.** With `PM1_DENSE=local`, the model is downloaded from HuggingFace on first use. After
   that it runs offline.
3. **HTTP services you configure** (`PM1_E5_*`, `PM1_PPLX_URL`, `PM1_RERANK_*`) receive queries and candidate texts.
   On the development machine they are all on `127.0.0.1`. Nothing is configured by default.
4. **Cursor judges.** `pm1-rule suggest` can ask two Cursor models (`composer-2.5`, `grok-4.7`) whether a candidate
   sentence is a reusable lesson, and whether a later message of yours changes a standing rule. That means the rule
   text and your message are sent, redacted. The judges only filter what is shown to you. They never activate or change
   anything. Every verdict is logged in `judge_log`. They run only if the external labeling tool `label_dual.py`
   exists at `PM1_LABEL_DUAL` (default `~/pm1-labeling/label_dual.py`, an external tool not included here, which a fresh install does not
   have). **Turn them off with `PM1_JUDGE=off`** or `pm1-rule suggest --no-judge`.
5. **The usage audit.** `pm1-meter audit` sends the previous day's real search queries and the records returned, redacted,
   to the same labeling tool (Cursor labelers, GPT on disagreements). **`PM1_JUDGE=off` stops the audit** (it is skipped, nothing is sent).
   `pm1-meter summary` and `pm1-meter dashboard` are local only.

## Background jobs

On the development machine the following run as systemd user units. Only `deploy/pm1-capture.service` (an example
unit using `%h` paths) and `deploy/install.sh --with-capture` are included in this repository; the other units were
specific to that machine and are described here for reference. Generated units are **(planned)**.

| Unit | Runs |
|---|---|
| `pm1-capture.service` | `pm1-capture watch -i 60` |
| `pm1-docs.service` | `pm1-docs watch --interval 300` |
| `pm1-backfill.timer` (01:00) | `pm1-worker --until 07:00 --batch 64 --rate 20 --newest-first --max-load 12 --exit-when-empty --report …` |
| `pm1-dashboard.timer` (hourly) | `pm1-meter dashboard` (writes `~/Documents/pm1-dashboard.html`, which must exist) |
| `pm1-meter.timer` (07:20) | `pm1-meter audit` + `dashboard` (see Privacy 5) |
| `pm1-lessons.timer` (07:40; unit files in `deploy/`) | `pm1-rule suggest --days 2 --max 5` |

## Configuration reference

Every setting is an environment variable. You can also put them in `~/.config/pm1/config.toml` (or the file named by
`PM1_CONFIG`): a flat `[pm1]` table where each key becomes `PM1_<KEY>`, for example

```toml
[pm1]
db = "~/pm1/memory.sqlite"   # PM1_DB
dense = "none"               # PM1_DENSE
judge = "off"                # PM1_JUDGE
```

An environment variable always wins over the file. Only documented settings are read from it (no program paths, and URLs only
to this machine); other keys are ignored. A missing or unreadable file is ignored.

| Variable | Default | Meaning |
|---|---|---|
| `PM1_DB` | `$XDG_STATE_HOME/pm1/memory.sqlite` | SQLite file |
| `PM1_BANK` | `personal` | namespace inside the DB; capture reads it once at import |
| `PM1_CLIENT` | empty | client name in the usage log |
| `PM1_DENSE` | `local` if fastembed is installed, else `none` | `local` (fastembed CPU) · `http` (`PM1_E5_URL`, default `127.0.0.1:8767`) · `split` (`PM1_E5_DOC_URL` / `PM1_E5_QUERY_URL`) · `none` |
| `PM1_TOKEN` | `none` | `pplx` (`PM1_PPLX_URL`, default `127.0.0.1:8770`) · `none` |
| `PM1_INDEX_MODELS` | derived from `PM1_DENSE` / `PM1_TOKEN` | which vector jobs capture and docs queue |
| `PM1_THREADS` | 8 | ONNX threads (local e5) |
| `PM1_RERANK_QWEN_VLLM`, `PM1_RERANK_QWEN`, `PM1_RERANK_GTE`, `PM1_RERANK_QWEN_CPU`, `PM1_RERANK_BGE_CPU` | unset | reranker service URLs |
| `PM1_RERANK_COLBERT` | unset | fastembed ColBERT model name (tries CUDA, falls back to CPU) |
| `PM1_USER_FIRST` | `0` | `1` = within the reranked head, the user's own words before the assistant's (questions about what the assistant said keep the plain order). Helps chat-style memory (BEAM-1M, LongMemEval); hurt on agent-work data, so it is off |
| `PM1_RERANK_K` | 16 | how many fused candidates the reranker reorders (the rest keep their fused order) |
| `PM1_RERANK_PROFILE` | `balanced` | default latency budget: `fast` 250 ms · `balanced` 1 s · `wait` 8 s · `best` unbounded |
| `PM1_QUALITY_FLOOR` | unset | agents cannot ask for a cheaper profile than this |
| `PM1_NEIGHBORS` | `5,2` | the top 5 hits bring their session neighbours (±2 turns, 600-char excerpts); `0` turns it off |
| `PM1_FACTS` | `0` | `1` = recall uses the fact index: turns pointed to by the facts nearest the question go near the top (LoCoMo +3 points over plain recall). Needs facts from `python -m pm1.facts_job` |
| `PM1_FACTS_URL`, `PM1_FACTS_MODEL` | unset, `qwen35-4b` | `pm1.facts_job`: OpenAI-compatible endpoint that writes the facts (use a model on this machine for real data); unset = the job does nothing |
| `PM1_FACT_HOP` | `0` | extra facts sharing an entity with the question (budgets ≥ 3,000); tied on LoCoMo, so off |
| `PM1_FACT_MIN_SIM` | `0` | facts less similar to the question than this pull in nothing |
| `PM1_FACT_GRAPH`, `PM1_FACT_GRAPH_MAX` | `0`, `100` | graph recall: hops over the fact graph (shared rare names + nearest facts); the reached facts' turns join the candidates before the rerank, so raise `PM1_RERANK_K` too. 0 = off |
| `PM1_DOCS`, `PM1_SESSIONS`, `PM1_CODE` | `1` | tool groups on/off |
| `PM1_CAPTURE_CURSOR` | `1` | capture Cursor chats |
| `PM1_SCOPE`, `PM1_DOC_SCOPE` | `default`, `docs` | default scopes |
| `PM1_TOOLOUT_DIR` | `$XDG_STATE_HOME/pm1/toolout` | hook: saved long outputs |
| `PM1_HOOK_TOOLOUT_LIMIT` | 16000 | hook: chars before compressing |
| `PM1_JUDGE` | `cursor` (active only if the labeling tool exists) | `off` disables the judges |
| `PM1_LABEL_DUAL`, `PM1_JUDGE_DIR`, `PM1_LABEL_DIR` | `~/pm1-labeling/label_dual.py`, `~/pm1-labeling` | labeling tool and its work folder (judges / meter) |

## Known limitations

- Linux only so far. The Cursor capture path (`~/.config/Cursor/User`) is Linux's, and macOS/Windows paths are **(planned)**.
  `CODEX_HOME` / `CLAUDE_CONFIG_DIR` are not honoured.
- graphify graphs are found only under the `PM1_GRAPH_GLOBS` locations.
- `pm1-meter audit` needs the external labeling tool; with `PM1_JUDGE=off` or without it, the audit is skipped and nothing is sent anywhere.
- The research modules in `src/pm1/` (`tree*`, `assert_mcp`, `memfs_mcp`, `ledger`, `graph`, `assertions`, `bgem3`,
  `llama_colbert`) are evaluation tools for the memory-structure experiments. They are not part of the product
  surface above.
