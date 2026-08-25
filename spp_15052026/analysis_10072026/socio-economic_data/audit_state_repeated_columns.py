#!/usr/bin/env python3
"""Audit columns that repeat across a *subset* of constituencies (e.g. every
constituency inside one city/Kreis sharing one district-level statistic) --
not just columns that happen to be uniform across an entire state.

A whole-state check misses this: Berlin/Hamburg are single-state city-states so
a shared district value looks "state-wide," but the same district-level-stat
artifact also shows up inside ordinary states wherever one city (Munich, Koln,
Essen, Dortmund, Duisburg, Dusseldorf, Stuttgart, Frankfurt, Leipzig, Hannover,
...) is split into multiple Bundestag constituencies -- the rest of that state
varies normally, so a "repeated in this state?" boolean never fires for them.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT = Path(__file__).with_name("socioeconomic_eng_clean.csv")
DEFAULT_OUTPUT = Path(__file__).with_name("state_repeated_columns_report.csv")
TOTAL_ROW_NAMES = {"Land insgesamt", "Insgesamt"}
ID_COLUMNS = {"state", "constituency_code", "constituency"}


def find_duplicate_clusters(rows: list[dict[str, str]], column: str) -> list[dict]:
    """Within each state, group constituencies sharing an identical non-blank
    value for `column`; any group of >=2 constituencies is a duplicate cluster
    (most likely a district-level stat duplicated across the constituencies
    that split that district), regardless of whether the rest of the state
    varies normally."""
    by_state: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_state[row["state"]].append(row)

    clusters = []
    for state, state_rows in by_state.items():
        by_value: dict[str, list[str]] = defaultdict(list)
        for row in state_rows:
            value = row[column]
            if value != "":
                by_value[value].append(row["constituency"])

        for value, constituencies in by_value.items():
            if len(constituencies) > 1:
                clusters.append(
                    {"state": state, "value": value, "constituencies": sorted(constituencies)}
                )

    return clusters


# This function audits the input CSV file for columns whose values repeat
# across two or more constituencies within the same state.
def audit(input_path: Path, output_path: Path) -> None:
    with input_path.open(newline="", encoding="utf-8") as source:
        rows = [
            row
            for row in csv.DictReader(source)
            if row["constituency"] not in TOTAL_ROW_NAMES
        ]

    if not rows:
        raise ValueError(f"No constituency rows found in {input_path}")

    columns = [name for name in rows[0] if name not in ID_COLUMNS]
    report_rows = []
    for column in columns:
        clusters = find_duplicate_clusters(rows, column)
        n_constituencies_affected = sum(len(c["constituencies"]) for c in clusters)
        affected_states = sorted({c["state"] for c in clusters})
        cluster_detail = " | ".join(
            f"{c['state']} [{len(c['constituencies'])}]: {', '.join(c['constituencies'])}"
            for c in sorted(clusters, key=lambda c: (-len(c["constituencies"]), c["state"]))
        )

        report_rows.append(
            {
                "column": column,
                "n_constituencies_affected": str(n_constituencies_affected),
                "n_clusters": str(len(clusters)),
                "affected_states": "; ".join(affected_states),
                "cluster_detail": cluster_detail,
            }
        )

    report_rows.sort(
        key=lambda row: (-int(row["n_constituencies_affected"]), row["column"])
    )

    with output_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find columns whose constituency values are repeated within states."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input cleaned federal CSV. Defaults to {DEFAULT_INPUT.name}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output audit CSV. Defaults to {DEFAULT_OUTPUT.name}.",
    )
    args = parser.parse_args()

    audit(args.input, args.output)
    print(f"Wrote repeated-column report to {args.output}")


if __name__ == "__main__":
    main()
