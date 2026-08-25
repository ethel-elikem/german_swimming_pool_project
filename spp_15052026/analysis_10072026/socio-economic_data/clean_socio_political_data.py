#!/usr/bin/env python3
"""Rebuild socio_political_data.csv with cleaned political columns (party shares,
CDU/CSU merged) instead of the raw per-party vote counts.

Reuses merge_political_data.py's load_political_rows() so the party grouping and
column naming stay identical to federal_baseline_eng_selected_political.csv.
"""
from pathlib import Path

import pandas as pd

from merge_political_data import POLITICAL_COLUMNS, load_political_rows

HERE = Path(__file__).resolve().parent
SOCIO_INPUT = HERE / "socioeconomic_eng_clean.csv"
POLITICAL_INPUT = HERE / "votes_share_raw.csv"
OUTPUT = HERE / "socio_political_data.csv"


def main() -> None:
    socio = pd.read_csv(SOCIO_INPUT)

    political_by_constituency = load_political_rows(POLITICAL_INPUT)
    political = pd.DataFrame.from_dict(political_by_constituency, orient="index")
    political.index = political.index.astype(int)
    political.index.name = "constituency_code"
    political = political.reset_index()
    for col in POLITICAL_COLUMNS:
        if col != "direct_mandate_party":
            political[col] = pd.to_numeric(political[col], errors="coerce")

    socio_political = socio.merge(political, on="constituency_code", how="left")
    missing = socio_political.loc[socio_political["valid_votes"].isna(), "constituency_code"].tolist()
    if missing:
        raise ValueError(f"missing political data for constituency_code: {missing}")

    socio_political.to_csv(OUTPUT, index=False)
    print(f"wrote {len(socio_political)} rows x {socio_political.shape[1]} cols to {OUTPUT.name}")
    print("political columns now:", [c for c in POLITICAL_COLUMNS])


if __name__ == "__main__":
    main()
