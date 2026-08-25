#!/usr/bin/env python3
"""Merge selected socio-economic constituency data with Bundestag 2025 results."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
DEFAULT_SOCIO_INPUT = HERE / "federal_baseline_eng_selected.csv"
DEFAULT_POLITICAL_INPUT = (
    PROJECT_ROOT / "SOCIAL_INTERACTIONS_2025/voters_share - second_vote_constituency.csv"
)
DEFAULT_OUTPUT = HERE / "federal_baseline_eng_selected_political.csv"

PARTY_COLUMNS = {
    "spd": ("SPD",),
    "cdu_csu": ("CDU", "CSU"),
    "greens": ("GREENS",),
    "fdp": ("FDP",),
    "afd": ("AfD",),
    "left": ("The Left",),
    "bsw": ("BSW",),
    "free_voters": ("FREE VOTERS",),
}
POLITICAL_COLUMNS = (
    "direct_mandate_party",
    "eligible_voters",
    "voters",
    "turnout_pct",
    "valid_votes",
    "invalid_votes",
    "invalid_votes_pct",
    "second_vote_spd_pct",
    "second_vote_cdu_csu_pct",
    "second_vote_greens_pct",
    "second_vote_fdp_pct",
    "second_vote_afd_pct",
    "second_vote_left_pct",
    "second_vote_bsw_pct",
    "second_vote_free_voters_pct",
)


def parse_int(value: str) -> int:
    value = value.strip()
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        return int(Decimal(value.replace(",", ".")))


def format_pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return ""
    value = (Decimal(numerator) / Decimal(denominator) * Decimal("100")).quantize(
        Decimal("0.01")
    )
    return format(value.normalize(), "f")


def load_political_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))

    by_constituency: dict[str, dict[str, str]] = {}
    for row in rows:
        constituency_code = row["constituency_code"]
        valid_votes = parse_int(row["valid_votes"])
        eligible_voters = parse_int(row["eligible_voters"])
        voters = parse_int(row["voters"])
        invalid_votes = parse_int(row["invalid_votes"])

        selected = {
            "direct_mandate_party": "" if row["chosen"] == "–" else row["chosen"],
            "eligible_voters": str(eligible_voters),
            "voters": str(voters),
            "turnout_pct": format_pct(voters, eligible_voters),
            "valid_votes": str(valid_votes),
            "invalid_votes": str(invalid_votes),
            "invalid_votes_pct": format_pct(invalid_votes, voters),
        }

        for output_name, source_names in PARTY_COLUMNS.items():
            votes = sum(parse_int(row.get(source_name, "")) for source_name in source_names)
            selected[f"second_vote_{output_name}"] = str(votes)
            selected[f"second_vote_{output_name}_pct"] = format_pct(votes, valid_votes)

        by_constituency[constituency_code] = {
            key: value for key, value in selected.items() if key in POLITICAL_COLUMNS
        }

    return by_constituency


def merge_data(socio_input: Path, political_input: Path, output: Path) -> None:
    political_by_constituency = load_political_rows(political_input)

    with socio_input.open(newline="", encoding="utf-8") as socio_source:
        socio_reader = csv.DictReader(socio_source)
        if socio_reader.fieldnames is None:
            raise ValueError(f"{socio_input} has no header")

        output_header = list(socio_reader.fieldnames) + list(POLITICAL_COLUMNS)
        missing_political_rows = []

        with output.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=output_header)
            writer.writeheader()

            for row in socio_reader:
                political = political_by_constituency.get(row["constituency_code"])
                if political is None:
                    missing_political_rows.append(row["constituency_code"])
                    political = {name: "" for name in POLITICAL_COLUMNS}

                writer.writerow(row | political)

    if missing_political_rows:
        missing = ", ".join(missing_political_rows)
        raise ValueError(f"Missing political data for constituency_code: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge selected socio-economic constituency data with selected Bundestag 2025 political columns."
    )
    parser.add_argument(
        "--socio-input",
        type=Path,
        default=DEFAULT_SOCIO_INPUT,
        help=f"Selected socio-economic CSV. Defaults to {DEFAULT_SOCIO_INPUT.name}.",
    )
    parser.add_argument(
        "--political-input",
        type=Path,
        default=DEFAULT_POLITICAL_INPUT,
        help="Bundestag 2025 second-vote constituency CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Merged output CSV. Defaults to {DEFAULT_OUTPUT.name}.",
    )
    args = parser.parse_args()

    merge_data(args.socio_input, args.political_input, args.output)
    print(f"Wrote merged CSV to {args.output}")


if __name__ == "__main__":
    main()
