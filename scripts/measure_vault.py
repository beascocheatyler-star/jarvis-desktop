#!/usr/bin/env python3
"""Measure the scope gate against your own vault and print a tuning grid.

Runs entirely offline — no API key, no tokens spent. Retrieval and the gate are
local, so this is free to run as often as you like.

    python scripts/measure_vault.py --vault ~/path/to/vault

Edit IN_SCOPE and OFF_TOPIC below to match your vault before trusting the
output. The defaults describe one particular vault and mean nothing for yours.

See docs/vault-tuning.md for the reasoning behind the shipped defaults.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.embedding import build_embedder  # noqa: E402
from app.core.guard import classify  # noqa: E402
from app.core.store import DocumentStore  # noqa: E402
from desktop.vault import VaultIndexer  # noqa: E402

# Questions whose answers really are somewhere in the vault.
IN_SCOPE: List[str] = [
    "what are my rules of engagement",
    "when is my next exam",
    "what is my GPA situation",
    "what are my daily tasks",
    "summarize my resume",
    "what is the Heuer Doctrine",
    "what is the Bunker Protocol",
    "what is my academic clearance plan",
    "what does my command center say",
    "what philosophy notes do I have",
    "what is on my canvas",
    "MATH3323 exam study guide",
    "CS1342 triage",
    "what are my projects",
    "what is the Nathan Override",
    "what licence do my projects use",
    "what is my physics test review",
]

# Questions that are plainly nothing to do with the vault.
OFF_TOPIC: List[str] = [
    "what is the capital of France",
    "how do I bake sourdough bread",
    "who won the 2022 World Cup",
    "what is the boiling point of mercury",
    "write me a poem about dragons",
    "how tall is Mount Everest",
    "what is the population of Tokyo",
    "how do I change a car tire",
    "what year did the Titanic sink",
    "explain quantum chromodynamics",
    "best recipe for carbonara",
    "how do I knit a scarf",
    "what is the airspeed velocity of a swallow",
    "who directed Jaws",
    "how do I train a parrot",
]

FLOORS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
RESCUES = [0.25, 0.30, 0.35, 1.00]


def percentiles(values: Sequence[float]) -> Tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    ordered = sorted(values)
    return (ordered[0], ordered[len(ordered) // 2], ordered[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", required=True, help="Path to the vault folder")
    parser.add_argument("--backend", default="hashing", help="Embedding backend")
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    logging.getLogger("pypdf").setLevel(logging.ERROR)

    root = Path(args.vault).expanduser().resolve()
    if not root.is_dir():
        print("Not a folder: {}".format(root), file=sys.stderr)
        return 1

    # A throwaway index so measuring never disturbs the app's own corpus.
    with tempfile.TemporaryDirectory() as tmp:
        store = DocumentStore(
            data_dir=Path(tmp) / "index",
            embedder=build_embedder(args.backend, args.dimensions),
        )
        report = VaultIndexer(store, root).scan()
        print("Vault: {}".format(root))
        print("Indexed: {}  ({} passages)".format(report.summary(), store.total_chunks))
        if store.is_empty:
            print("Nothing indexed; cannot measure.", file=sys.stderr)
            return 1

        hits = {q: store.index.search(q, top_k=args.top_k) for q in IN_SCOPE + OFF_TOPIC}
        matched = store.count_matching(None)

        def gate(q: str, floor: float, rescue: float) -> bool:
            return classify(
                question=q,
                hits=hits[q],
                relevance_floor=floor,
                corpus_empty=False,
                scope=None,
                scope_matched_documents=matched,
                dense_rescue_floor=rescue,
            ).allows_llm

        import math

        def lexical(q: str) -> float:
            h = hits[q]
            if not h:
                return 0.0
            return math.sqrt(max(h[0].normalized, 0.0) * max(h[0].coverage, 0.0))

        def dense(q: str) -> float:
            return max((x.dense_score for x in hits[q]), default=0.0)

        print("\n=== Distributions (min / median / max) ===")
        for label, qs in (("In-scope ", IN_SCOPE), ("Off-topic", OFF_TOPIC)):
            lo, mid, hi = percentiles([lexical(q) for q in qs])
            dlo, dmid, dhi = percentiles([dense(q) for q in qs])
            print(
                "{}  lexical {:.3f} / {:.3f} / {:.3f}   dense {:.3f} / {:.3f} / {:.3f}".format(
                    label, lo, mid, hi, dlo, dmid, dhi
                )
            )

        print("\n=== Grid: false refusals (/{}) / false answers (/{}) ===".format(
            len(IN_SCOPE), len(OFF_TOPIC)
        ))
        header = "floor  " + "".join("r={:<8.2f}".format(r) for r in RESCUES)
        print(header)
        for floor in FLOORS:
            row = "{:.2f}   ".format(floor)
            for rescue in RESCUES:
                refusals = sum(1 for q in IN_SCOPE if not gate(q, floor, rescue))
                answers = sum(1 for q in OFF_TOPIC if gate(q, floor, rescue))
                row += "{:>2}/{:<7}".format(refusals, answers)
            print(row)

        print("\n=== Questions refused at the shipped defaults (0.45 / 0.35) ===")
        for q in IN_SCOPE:
            if not gate(q, 0.45, 0.35):
                print("  refused:  {}  (lexical {:.2f})".format(q, lexical(q)))
        for q in OFF_TOPIC:
            if gate(q, 0.45, 0.35):
                print("  answered: {}  (lexical {:.2f}, dense {:.2f})".format(
                    q, lexical(q), dense(q)
                ))

        print(
            "\nA question whose answer is genuinely absent from the vault is not\n"
            "a false refusal — the gate is right and the vault is incomplete."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
