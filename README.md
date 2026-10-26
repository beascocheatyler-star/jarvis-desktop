# Jarvis — your vault, answered

A Mac app that answers questions from an Obsidian vault. Hybrid retrieval
(BM25 + dense vectors, fused with Reciprocal Rank Fusion) and a deterministic
scope gate run locally; Claude is called only when a question passes the gate.

Built on [hybrid-scoped-rag](https://github.com/beascocheatyler-star/hybrid-scoped-rag).
The retrieval core, scope gate, and fusion logic are that project's, unchanged.
This adds a vault indexer, a desktop UI, and defaults measured against a real
vault rather than the bundled demo corpora.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python jarvis.py
```

No server, no port, no browser. One window.

---

## What it does

Point it at a folder. It indexes every note, then answers questions about them
and acts on them — searching, reading, creating, and editing notes the way a
person would.

### Two modes

The toggle under the composer switches between them.

**Agent** (default) gives Claude seventeen tools across four groups:

| Group | Tools |
|---|---|
| Vault | `search_vault`, `read_note`, `list_notes`, `list_folders`, `create_note`, `append_to_note`, `write_note`, `create_folder`, `delete_note` |
| Machine | `read_file`, `write_file`, `edit_file`, `list_directory`, `find_files`, `search_files`, `open_path` |
| Shell | `run_command` |
| Web | `web_search` |

It can read and write anywhere on the Mac, run tests and git, search a
codebase, and look things up online — the same reach a developer has at a
terminal. Give it a messy description of a project and it will plan, build,
run the tests, and report what actually passed.

**Ask** is the single-pass retrieve-and-answer path: one API call, no tools,
cannot modify anything. Cheaper, and the scope gate refuses off-topic
questions before spending a token.

### Approval

| Action | Prompts? |
|---|---|
| Reading, searching, listing, web search | Never |
| Creating a note, appending, creating a folder | Never — these cannot destroy anything |
| Overwriting or deleting a **note** | Yes, unless you turn it off in Settings |
| Any **shell command**, any write **outside the vault** | **Always** — cannot be turned off |

That last row is deliberate. "Stop asking about my own notes" is a reasonable
preference; "run anything on my Mac without telling me" is a different
decision, so the auto-approve setting does not extend to it. The approval
dialog shows the exact command or the full file content before you decide.

A short list of unrecoverable commands is refused outright regardless of
approval — `rm -rf /`, writing to a raw device, fork bombs, piping a download
into a shell. The loop is capped at 40 tool round-trips.

### Browsing and editing

The sidebar is a real file tree. Each sector row has two controls:

| Control | Action |
|---|---|
| **›** arrow / sector name | Expand to see the notes inside |
| **○** circle on the right | Limit search to that sector |

They are deliberately separate — looking inside a folder should not silently
change what the assistant can see.

Click a note to read it rendered. The toolbar above it gives you edit
(**⌘E**), save (**⌘S**), reveal in Finder, and delete. Saving re-indexes
immediately, so the assistant can answer from the new text straight away.
Citations under an answer are clickable: they open the note they came from.

Create notes with **⌘N** and folders with **⇧⌘N**. A new note opens straight
into edit mode. Notes you edit in Obsidian are picked up automatically about a
second after you save, and if that note is open in Jarvis the view refreshes —
unless you are mid-edit, where reloading would discard your typing.

An empty folder will not appear in the tree until it holds a note; the tree is
built from indexed documents, not from the directory listing.

### Scoping

Your folder structure becomes the search scope automatically:
`02-Academy/canvas.md` is tagged `sector: academy`, so scoping to **Academy**
restricts retrieval to those notes. YAML frontmatter is lifted into the same
metadata, so `course: CS1342` becomes filterable for free.

## Built around how the app is actually used

- **It opens where you left off.** Last vault reloads automatically, cursor
  already in the input. No wizard, no config screen, nothing between launch
  and typing.
- **Status is a sentence.** "Indexing 24/32 · canvas.md", then "Searching 32
  notes…", then "Writing…". You can always tell what is happening and whether
  it is stuck.
- **The transcript scrolls properly.** Trackpad and scroll wheel work through
  the full history. New messages auto-follow, but scrolling up to re-read
  suspends that so nothing yanks the view mid-sentence — a jump-to-latest
  button brings you back.
- **Sources are collapsed by default.** The answer is the answer. Citations,
  retrieval scores, and which retriever found what are one click away when you
  want to audit them, invisible when you don't.
- **Refusals look different from errors.** A gate refusal is a normal,
  zero-cost outcome and is styled as one, with the score that caused it shown.
- **Edits are picked up automatically.** Save a note in Obsidian and it is
  re-indexed about a second later. No manual refresh.

Shortcuts: `⌘N` new note · `⇧⌘N` new folder · `⌘T` new chat · `⌘E` edit ·
`⌘S` save · `⌘R` rescan · `⌘K` focus input · `⌘,` settings · `Esc` close note ·
`Enter` send · `Shift+Enter` newline.

## Costs and privacy

Indexing, retrieval, scoping, and the gate are local and free — they need no
API key and no network. Only generation calls Claude.

Agent mode costs more than Ask mode because it makes several calls per
request: roughly $0.10–0.20 for a real task on Opus 5, against one call for
a plain question. Switching the model to Sonnet in Settings cuts that by
about 5x. Every answer shows its own cost.

Your notes are never uploaded anywhere. Retrieved passages are sent to the
Anthropic API as context when a question is answered; nothing else leaves the
machine. The API key is stored in the macOS Keychain, not in a config file
inside your vault — an iCloud-synced plaintext key would replicate to every
device you own.

Two folders are skipped by default: `copilot/` (past AI chat transcripts) and
`outputs/` (generated files). Indexing old chat logs makes the assistant cite
its own earlier guesses back as if they were source notes. On the vault this
was built against, they were 47 of 85 documents. Change this in Settings.

## Tuning

The shipped defaults — `relevance_floor 0.45`, `dense_rescue_floor 0.35` —
were measured on a real 32-note vault, not inherited from upstream. The
upstream defaults produced 6 false answers out of 15 off-topic questions here,
because the hashing embedder scores generic question phrasing highly
regardless of topic.

Measure your own vault:

```bash
python scripts/measure_vault.py --vault ~/path/to/vault
```

Edit the question lists at the top of that script first. Full reasoning and
the measured distributions are in [`docs/vault-tuning.md`](docs/vault-tuning.md).

For real semantic matching, `pip install sentence-transformers` and set the
backend in Settings — then re-measure, because the thresholds will not carry
over.

## Build a double-clickable app

```bash
./build_app.sh
```

Produces `dist/Jarvis.app`. Drag it to Applications.

## Tests

```bash
pytest
```

393 tests, all offline, no API key required (the agent's API client is
stubbed). The upstream suite plus tests for the vault indexer, the agent, and
the machine tools. Several are regressions for bugs found during the build:
a prune step that deleted 29 of 32 indexed notes on the second scan, an
agent token budget so small it split one source file into three to work
around it, and an API failure mid-build that escaped as a traceback instead
of a readable message.

## Layout

```
app/              retrieval, gate, fusion, store — upstream, unchanged
desktop/
  vault.py        folder -> DocumentStore, metadata from folders + frontmatter
  controller.py   in-process replacement for the FastAPI layer
  ui.py           Flet window
  watcher.py      debounced re-index on file change
  appsettings.py  preferences + Keychain
  agent.py        tool loop, approval gate, step limit, API error handling
  tools.py        vault tools + registration of the machine tools
  system_tools.py shell, files anywhere, grep/glob, open — with the refusal list
scripts/
  measure_vault.py  re-tune the gate against your own vault
docs/
  vault-tuning.md   measured distributions and threshold reasoning
```

## Licence

GPL-3.0. Copyright © 2026 Tyler Beascochea. Derivative works must also be
GPL-3.0 with source available. See [LICENSE](LICENSE).

Upstream `hybrid-scoped-rag` is also GPL-3.0. Flet is Apache-2.0; `anthropic`,
`pypdf`, and `watchdog` carry their own licences, unaffected by this one.
