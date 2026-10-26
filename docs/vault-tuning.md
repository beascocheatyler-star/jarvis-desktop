# Scope-gate tuning for a personal vault

The upstream defaults (`RELEVANCE_FLOOR=0.40`, `DENSE_RESCUE_FLOOR=0.25`) were
measured against the bundled course-syllabus and customer-support corpora. A
personal Obsidian vault is a different distribution, and the defaults do not
transfer. This document records what was actually measured and why the desktop
app ships different numbers.

Re-run the measurement against your own vault with `scripts/measure_vault.py`.

## Method

32 notes (153 passages) from a real vault, `hashing` embedder, 1024
dimensions, top-6. 17 questions whose answers are genuinely in the vault, 15
that are plainly off-topic. Reproduce with:

```bash
python scripts/measure_vault.py --vault ~/path/to/vault
```

## Measured distributions

| | min | median | max |
|---|---|---|---|
| In-scope lexical | 0.451 | 0.672 | 0.908 |
| Off-topic lexical | 0.000 | 0.299 | 0.510 |
| In-scope dense | 0.168 | 0.254 | 0.493 |
| Off-topic dense | 0.119 | 0.179 | 0.317 |

These are measured *after* filename indexing (see below). Before it, in-scope
lexical bottomed out at 0.369 and two real questions fell under the floor.

The lexical signal separates the two populations well. **The dense signal does
not.** Off-topic dense (max 0.317) exceeds in-scope dense median (0.247):
"what is the population of Tokyo" scores 0.317 against this vault while
"summarize my resume" scores 0.107.

This is a property of the hashing embedder, not a bug. It hashes character
n-grams, so it is measuring shared surface form — generic English question
phrasing — rather than meaning. Every question shares that form, so the score
tracks question-shapedness and not topic.

## Grid

False refusals out of 17 in-scope / false answers out of 15 off-topic:

| floor | r=0.25 | r=0.30 | r=0.35 | r=1.00 (off) |
|---|---|---|---|---|
| 0.30 | 0 / 9 | 0 / 7 | 0 / 7 | 0 / 7 |
| 0.35 | 0 / 8 | 0 / 6 | 0 / 6 | 0 / 6 |
| 0.40 | 0 / 6 | 0 / 4 | 0 / 4 | 0 / 4 |
| **0.45** | 0 / 6 | 0 / 3 | **0 / 2** | 0 / 2 |
| 0.50 | 1 / 6 | 1 / 3 | 1 / 2 | 1 / 2 |
| 0.55 | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 0 |

Two things are visible:

1. **The dense rescue never helps on this corpus.** At every floor, lowering
   the rescue threshold adds false answers and removes no false refusals.
   Since off-topic dense scores overlap the in-scope range entirely, any
   rescue threshold low enough to admit a real question also admits several
   off-topic ones. `r=0.35` sits at the point where the rescue stops firing
   for practical purposes; `r=1.00` disables it outright and is never worse.
2. **Strictness is not free.** Going from 0.45 to 0.55 eliminates the last two
   false answers but starts refusing real questions.

## Filename indexing

`Chunk` tokenises only its `text`; `doc_name` is carried for citations but
never indexed. A note called `canvas.md` was therefore unfindable by the word
"canvas" unless its body happened to repeat it — "what is on my canvas"
retrieved three unrelated triage notes and scored 0.31, under the floor.

`desktop/vault.py` now prepends a title header to each text note before
indexing:

```
# canvas Academy

(Note: 02-Academy/canvas.md)
```

Separators are split and letter/digit runs broken apart, so `MATH3323_Exam2`
yields `MATH 3323 Exam 2` and the bare course number matches. The header is
skipped when the note already opens with a heading containing the same words,
and never applied to PDF or DOCX, whose bytes are parsed structurally.

Effect on the same question set: the "canvas" question went 0.31 → 0.91, the
in-scope lexical minimum rose 0.369 → 0.451, and **false refusals fell from 2
to 0** at the shipped defaults.

## Chosen defaults

`relevance_floor = 0.45`, `dense_rescue_floor = 0.35`.

0 false refusals, 2 false answers. A false answer costs one guarded API call
where the model is explicitly told to verify that the context answers the
question — it usually then says it cannot. A false refusal costs a real
question going unanswered and the user having to rephrase and retry. For a tool
used dozens of times a day, the interruption is the more expensive failure, so
the balanced point is preferred over the strict one.

## If you switch to sentence-transformers

Re-measure. True semantic embeddings should separate the dense distributions
properly, at which point the rescue becomes useful again and the floor can
likely come down. The rescue threshold will need to be substantially higher
(~0.5) because cosine similarities from a real embedder are not on the same
scale as hashing similarities.

```bash
pip install sentence-transformers
# set EMBEDDING_BACKEND=sentence-transformers, then:
python scripts/measure_vault.py --vault ~/path/to/vault
```

## Re-tuning for your own vault

Edit the question lists at the top of `scripts/measure_vault.py` — 15–20 real
questions you would actually ask, and 15 clearly off-topic ones — then run it.
The script prints the distributions and the grid above. Pick the cell whose
trade-off you prefer and set it in Settings, or in
`~/Library/Application Support/JarvisDesktop/settings.json`.

Note that a question whose answer is genuinely *not* in your vault is not a
false refusal. "What classes am I taking" is refused here because no note
actually lists current enrolment — the gate is correct and the vault is
incomplete.
