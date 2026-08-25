#!/usr/bin/env python3
"""Clean constituency-level columns from socioeconomic_eng_raw.csv.

Last updated: 2026-08-13.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path
import csv


DEFAULT_INPUT = Path(__file__).with_name("socioeconomic_eng_raw.csv")
DEFAULT_OUTPUT = Path(__file__).with_name("socioeconomic_eng_clean.csv")

DATE_TEXT_TO_DROP = "as of 31.12.2023"
SHORT_COLUMN_NAMES = {
    "state": "state",
    "Municipalities (number)": "municipalities",
    "Area (km²)": "area_km2",
    "Population – Total (number)": "total_population",
    "Population – Germans (number)": "german_population",
    "Population – Foreign nationals (%)": "foreign_population_pct",
    "Population density (inhabitants per km²)": "population_density_per_km2",
    "Population increase (+) or decrease (-) 2023 – Natural birth balance (per 1,000 inhabitants)": "natural_population_balance_per_1k",
    "Population increase (+) or decrease (-) 2022 – Net migration balance (per 1,000 inhabitants)": "net_migration_balance_per_1k",
    "Age group – under 18 (%)": "age_under_18_pct",
    "Age group – 18–24 (%)": "age_18_24_pct",
    "Age group – 25–34 (%)": "age_25_34_pct",
    "Age group – 35–59 (%)": "age_35_59_pct",
    "Age group – 60–74 (%)": "age_60_74_pct",
    "Age group – 75 and over (%)": "age_75_plus_pct",
    "Land area by actual use as of 31.12.2022 – Settlement and transport (%)": "settlement_transport_land_pct",
    "Land area by actual use as of 31.12.2022 – Vegetation and water bodies (%)": "vegetation_water_land_pct",
    "Completed dwellings 2023 (per 1,000 inhabitants)": "completed_dwellings_per_1k",
    "Housing stock – total (per 1,000 inhabitants)": "housing_stock_per_1k",
    "Living space (per dwelling)": "living_space_per_dwelling",
    "Living space (per inhabitant)": "living_space_per_inhabitant",
    "Car stock as of 01.01.2024 – total cars (per 1,000 inhabitants)": "cars_per_1k",
    "Car stock as of 01.01.2024 – cars with electric or hybrid drive (%)": "electric_hybrid_cars_pct",
    "Business register 2022 – total enterprises (per 1,000 inhabitants)": "enterprises_per_1k",
    "Business register 2022 – skilled-trade (craft) enterprises (per 1,000 inhabitants)": "craft_enterprises_per_1k",
    "Graduates of vocational schools 2022": "vocational_school_graduates",
    "Graduates of general education schools 2022 – total, excluding external candidates (per 1,000 inhabitants)": "school_graduates_per_1k",
    "Graduates of general education schools 2022 – without lower secondary certificate (%)": "school_grads_no_lower_secondary_pct",
    "Graduates of general education schools 2022 – with lower secondary certificate (%)": "school_grads_lower_secondary_pct",
    "Graduates of general education schools 2022 – with intermediate secondary certificate": "school_grads_intermediate_secondary",
    "Graduates of general education schools 2022 – with general or applied higher education entrance qualification (%)": "school_grads_higher_education_entrance_pct",
    "Childcare as of 01.03.2023 – children under 3 in care (care rate)": "childcare_under_3_rate",
    "Childcare as of 01.03.2023 – children aged 3 to under 6 in care (care rate)": "childcare_age_3_5_rate",
    "Disposable income of private households 2021 (EUR per inhabitant)": "disposable_income_eur_per_inhabitant",
    "Gross domestic product 2021 (EUR per inhabitant)": "gdp_eur_per_inhabitant",
    "Employees subject to social insurance contributions as of 30.06.2023 – total (per 1,000 inhabitants)": "employees_per_1k",
    "Employees subject to social insurance contributions – Agriculture, forestry, fisheries (%)": "employees_agriculture_pct",
    "Employees subject to social insurance contributions – Manufacturing/production sector (%)": "employees_manufacturing_pct",
    "Employees subject to social insurance contributions – Trade, hospitality, transport (%)": "employees_trade_hospitality_transport_pct",
    "Employees subject to social insurance contributions – Public and private service providers (%)": "employees_public_private_services_pct",
    'Employees subject to social insurance contributions – Other service providers and "not specified" (%)': "employees_other_services_pct",
    "Recipients of benefits under SGB II (Social Code Book II) August 2024 – total (per 1,000 inhabitants)": "sgb_ii_recipients_per_1k",
    "Recipients of SGB II benefits – dependents unable to work (%)": "sgb_ii_unable_to_work_pct",
    "Recipients of SGB II benefits – foreign nationals (%)": "sgb_ii_foreign_nationals_pct",
    "Unemployment rate November 2024 – total": "unemployment_total_pct",
    "Unemployment rate November 2024 – men": "unemployment_men_pct",
    "Unemployment rate November 2024 – women": "unemployment_women_pct",
    "Unemployment rate November 2024 – ages 15 to 24": "unemployment_age_15_24_pct",
    "Unemployment rate November 2024 – ages 55 to 64": "unemployment_age_55_64_pct",
}
TOTAL_ROW_NAMES = {"Land insgesamt", "Insgesamt"}


def parse_number(value: str) -> Decimal | None:
    normalized = value.strip().replace(",", ".")
    if not normalized:
        return None

    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def format_number(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def clean_header(name: str) -> str:
    cleaned = name.replace(DATE_TEXT_TO_DROP, "")
    for marker in ("(in 1,000s)", "(in 1000s)", "(in 100s)"):
        cleaned = cleaned.replace(marker, "(number)")
    return " ".join(cleaned.split())


def is_thousand_column(name: str) -> bool:
    return any(marker in name for marker in ("(in 1,000s)", "(in 1000s)", "(in 100s)"))


def short_column_name(name: str) -> str:
    return SHORT_COLUMN_NAMES.get(name, name)


def clean_csv(input_path: Path, output_path: Path) -> None:
    with input_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        header = next(reader)
        thousand_column_indexes = {
            index for index, name in enumerate(header) if is_thousand_column(name)
        }
        cleaned_header = [clean_header(name) for name in header]
        final_header = [short_column_name(name) for name in cleaned_header]
        constituency_index = final_header.index("constituency")

        with output_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target)
            writer.writerow(final_header)

            for row in reader:
                if row[constituency_index] in TOTAL_ROW_NAMES:
                    continue

                cleaned_row = []
                for index, value in enumerate(row):
                    number = parse_number(value)
                    if number is None:
                        cleaned_row.append(value)
                        continue

                    if index in thousand_column_indexes:
                        number *= Decimal("1000")

                    cleaned_row.append(format_number(number))

                writer.writerow(cleaned_row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert German decimal commas to dots and shorten column names."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path. Defaults to {DEFAULT_INPUT.name}.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Defaults to {DEFAULT_OUTPUT.name}.",
    )
    args = parser.parse_args()

    clean_csv(args.input, args.output)
    print(f"Wrote cleaned CSV to {args.output}")


if __name__ == "__main__":
    main()
