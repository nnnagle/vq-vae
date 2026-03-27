"""
Combine NE and SE LANDFIRE EVT ecological systems contingency tables.

Both tables are square confusion matrices with LANDFIRE numeric codes as row
and column labels. Codes partially overlap between regions. The combined table
covers the union of all codes with overlapping cell counts summed.
"""

import pandas as pd
from pathlib import Path

SE_PATH = Path(__file__).parent / "SE_Remap_EVT_Agreeement_Assessment/SE_Remap_EcologicalSystems_ContingencyTable_clean.csv"
NE_PATH = Path(__file__).parent / "NE_Remap_EVT_Agreement_Assessment/NE_Remap_EcologicalSystems_ContingencyTable_clean.csv"
OUT_PATH = Path(__file__).parent / "combined_evt_contingency_table.csv"

SUMMARY_COLS = ["Row Totals", "Percent Row Agreement"]
SUMMARY_ROWS = ["Column Totals", "Percent Column Agreement"]


def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    # Drop summary columns
    cols_to_drop = [c for c in SUMMARY_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    # Drop summary rows and any NaN index rows
    rows_to_drop = [r for r in SUMMARY_ROWS if r in df.index]
    df = df.drop(index=rows_to_drop)
    df = df[df.index.notna()]
    # Ensure row and column labels are integers
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    return df.astype(float)


def combine(se: pd.DataFrame, ne: pd.DataFrame) -> pd.DataFrame:
    combined = se.add(ne, fill_value=0).fillna(0).astype(int)
    # Sort rows and columns numerically
    codes = sorted(combined.index)
    combined = combined.reindex(index=codes, columns=codes, fill_value=0)
    return combined


def add_summary_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Row Totals"] = df.sum(axis=1)
    diag = pd.Series(
        [df.at[code, code] if code in df.columns else 0 for code in df.index],
        index=df.index,
    )
    df["Percent Row Agreement"] = (diag / df["Row Totals"].replace(0, float("nan")) * 100).fillna(0.0)
    return df


def main():
    se = load_table(SE_PATH)
    ne = load_table(NE_PATH)

    print(f"SE table: {se.shape[0]} codes")
    print(f"NE table: {ne.shape[0]} codes")

    combined = combine(se, ne)
    print(f"Combined table: {combined.shape[0]} codes "
          f"({len(set(se.index) & set(ne.index))} overlapping)")

    combined = add_summary_cols(combined)
    combined.index.name = "LANDFIRE"
    combined.to_csv(OUT_PATH)
    print(f"Written to {OUT_PATH}")


if __name__ == "__main__":
    main()
