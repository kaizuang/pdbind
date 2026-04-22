"""Analyze LP-PDBBind dataset qualities.

Answers:
  1. How many protein-ligand structures per split and per protein type?
  2. Average protein size (sequence length) and ligand size (heavy atoms) per category?
  3. Average binding affinity (-log10 Kd/Ki, stored in the `value` column) per category?

"Category" is interpreted along three axes: `new_split` (train/val/test),
`type` (protein class), and `category` (PDBBind refined/general/core).

Ligand heavy-atom count uses RDKit when available; otherwise falls back to a
SMILES heuristic (uppercase atomic-symbol chars + bracketed atoms) which
overcounts slightly on aromatic lowercase tokens but is within a few percent.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "LP-PDBBind" / "dataset" / "LP_PDBBind.csv"


def ligand_heavy_atoms(smiles: str) -> int | None:
    if not isinstance(smiles, str) or not smiles:
        return None
    if _HAS_RDKIT:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol.GetNumHeavyAtoms()
    # Heuristic fallback: count heavy-atom tokens in SMILES.
    # Strip bracketed atoms first (each bracket = one heavy atom), then count
    # organic-subset atom tokens (Cl, Br, then single letters).
    bracket_atoms = len(re.findall(r"\[[^\]]+\]", smiles))
    stripped = re.sub(r"\[[^\]]+\]", "", smiles)
    two_letter = len(re.findall(r"Cl|Br", stripped))
    stripped = re.sub(r"Cl|Br", "", stripped)
    single = len(re.findall(r"[BCNOPSFIbcnops]", stripped))
    return bracket_atoms + two_letter + single


def protein_length(seq: str) -> int | None:
    if not isinstance(seq, str):
        return None
    return len(seq)


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df["protein_len"] = df["seq"].map(protein_length)
    df["ligand_heavy_atoms"] = df["smiles"].map(ligand_heavy_atoms)
    return df


def summarize(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    g = df.groupby(group_col, dropna=False)
    out = pd.DataFrame({
        "count": g.size(),
        "avg_protein_len": g["protein_len"].mean().round(1),
        "avg_ligand_heavy_atoms": g["ligand_heavy_atoms"].mean().round(2),
        "avg_affinity_value": g["value"].mean().round(3),
        "median_affinity_value": g["value"].median().round(3),
        "n_covalent": g["covalent"].sum().astype(int),
    })
    return out.sort_values("count", ascending=False)


def split_by_type(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="type", columns="new_split", values="smiles", aggfunc="count", fill_value=0
    )
    pivot["total"] = pivot.sum(axis=1)
    return pivot.sort_values("total", ascending=False)


def print_block(title: str, frame: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 160
    ):
        print(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV, help="Path to LP_PDBBind.csv"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory to write CSV summaries",
    )
    args = parser.parse_args()

    if not _HAS_RDKIT:
        print(
            "[warn] RDKit not available; using SMILES heuristic for ligand heavy-atom "
            "counts. Install rdkit for exact counts.\n"
        )

    df = load(args.csv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} complexes from {args.csv}")
    print(f"Columns: {list(df.columns)}")

    # Q1: counts per split, per protein type, and the split x type cross-tab
    split_counts = df["new_split"].value_counts().rename("count").to_frame()
    print_block("Q1a. Complexes per split (new_split)", split_counts)

    type_counts = df["type"].value_counts().rename("count").to_frame()
    print_block("Q1b. Complexes per protein type", type_counts)

    split_type = split_by_type(df)
    print_block("Q1c. Protein type x split", split_type)

    # Q2 + Q3: averages grouped along three axes
    by_split = summarize(df, "new_split")
    print_block("Q2/Q3 by split (new_split)", by_split)

    by_type = summarize(df, "type")
    print_block("Q2/Q3 by protein type", by_type)

    by_cat = summarize(df, "category")
    print_block("Q2/Q3 by PDBBind category (refined/general/core)", by_cat)

    # Also show clean-level breakdown since LP-PDBBind emphasizes CL1/2/3
    cl_rows = []
    for cl in ["CL1", "CL2", "CL3"]:
        sub = df[df[cl].astype(bool)]
        cl_rows.append({
            "clean_level": cl,
            "count": len(sub),
            "avg_protein_len": round(sub["protein_len"].mean(), 1),
            "avg_ligand_heavy_atoms": round(sub["ligand_heavy_atoms"].mean(), 2),
            "avg_affinity_value": round(sub["value"].mean(), 3),
        })
    cl_df = pd.DataFrame(cl_rows).set_index("clean_level")
    print_block("Q2/Q3 by clean level (CL1/CL2/CL3)", cl_df)

    # Persist CSVs for downstream use / report
    split_counts.to_csv(args.outdir / "counts_by_split.csv")
    type_counts.to_csv(args.outdir / "counts_by_type.csv")
    split_type.to_csv(args.outdir / "counts_type_x_split.csv")
    by_split.to_csv(args.outdir / "summary_by_split.csv")
    by_type.to_csv(args.outdir / "summary_by_type.csv")
    by_cat.to_csv(args.outdir / "summary_by_category.csv")
    cl_df.to_csv(args.outdir / "summary_by_clean_level.csv")

    print(f"\nWrote summary CSVs to {args.outdir}")


if __name__ == "__main__":
    main()
