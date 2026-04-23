"""Extract a single protein-ligand structure from the LP-PDBBind archives.

Usage:
    python extract_structure.py 7EJK
    python extract_structure.py 7ejk --out /tmp/structures
    python extract_structure.py 5EDQ --archive EGFR
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = REPO_ROOT / "LP-PDBBind" / "dataset"

ARCHIVES = ["BDB2020+", "mpro", "EGFR"]


def members_for(archive: str, pdb_id: str) -> list[str]:
    pid = pdb_id.upper()
    if archive == "BDB2020+":
        base = f"BDB2020+/dataset/{pid}"
        return [f"{base}/protein.pdb", f"{base}/ligand.pdb",
                f"{base}/ligand.sdf", f"{base}/pdb{pid.lower()}.ent"]
    if archive == "mpro":
        base = f"mpro/{pid}"
        return [f"{base}/protein.pdb", f"{base}/ligand.pdb",
                f"{base}/ligand.sdf", f"{base}/{pid}.pdb",
                f"{base}/protein_water.pdb"]
    if archive == "EGFR":
        return [f"EGFR/protein/{pid}.pdb",
                f"EGFR/ligand_addH/{pid}_with_H.sdf"]
    raise ValueError(f"unknown archive: {archive}")


def extract(pdb_id: str, out_dir: Path, archive: str | None = None) -> Path:
    archives = [archive] if archive else ARCHIVES
    for name in archives:
        tgz = ARCHIVE_DIR / f"{name}.tgz"
        if not tgz.exists():
            continue
        wanted = set(members_for(name, pdb_id))
        with tarfile.open(tgz, "r:gz") as tar:
            present = [m for m in tar.getmembers() if m.name in wanted]
            if not present:
                continue
            tar.extractall(out_dir, members=present)
        return out_dir
    raise FileNotFoundError(f"{pdb_id} not found in {archives}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdb_id")
    ap.add_argument("--out", type=Path, default=Path.cwd() / "extracted")
    ap.add_argument("--archive", choices=ARCHIVES)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out = extract(args.pdb_id, args.out, args.archive)
    print(f"Extracted {args.pdb_id.upper()} to {out}")


if __name__ == "__main__":
    main()
