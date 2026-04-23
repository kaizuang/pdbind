"""Compute residue-level adjacency matrices for one extracted structure.

Nodes are residues (standard amino acids from the first model, ordered by
appearance in the PDB). Two definitions of residue-residue distance are
supported:
  * CA   -- Euclidean distance between C-alpha atoms
  * MHAD -- minimum heavy-atom distance between the two residues

For each distance definition, four unweighted adjacency matrices are
produced at cutoffs tau in {3, 5} Angstrom:
  1) Euclidean residue distance, cutoff tau
  2) Geodesic (Isomap-style shortest path) on the neighborhood graph built
     at radius tau, cutoff tau

The geodesic edge weight uses the same residue distance. Spectral
decomposition of the graph Laplacian (Laplacian eigenmaps) is also saved for
downstream use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import laplacian, shortest_path
from scipy.spatial.distance import cdist

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def parse_residues(pdb_path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Return per-residue heavy-atom coordinates, per-residue CA coordinates
    (NaN if missing), and residue labels from the first MODEL block."""
    residues: dict[tuple, list[tuple[float, float, float]]] = {}
    ca: dict[tuple, tuple[float, float, float]] = {}
    order: list[tuple] = []
    seen_altloc: dict[tuple, set[str]] = {}

    with pdb_path.open() as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM"):
                continue
            altloc = line[16]
            resname = line[17:20].strip()
            if resname not in STANDARD_AA:
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip() or atom_name.lstrip("0123456789")[:1]
            if element == "H" or element == "D":
                continue
            chain = line[21]
            resseq = int(line[22:26])
            icode = line[26].strip()
            key = (chain, resseq, icode, resname)

            alts = seen_altloc.setdefault(key, set())
            if altloc != " " and altloc not in alts:
                if alts and altloc != "A":
                    continue
                alts.add(altloc)

            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            if key not in residues:
                residues[key] = []
                order.append(key)
            residues[key].append((x, y, z))
            if atom_name == "CA":
                ca[key] = (x, y, z)

    heavy = [np.asarray(residues[k], dtype=np.float32) for k in order]
    ca_coords = [np.asarray(ca.get(k, (np.nan, np.nan, np.nan)),
                            dtype=np.float32) for k in order]
    labels = [f"{k[0]}:{k[3]}{k[1]}{k[2]}" for k in order]
    return heavy, ca_coords, labels


def mhad_distance(coords: list[np.ndarray]) -> np.ndarray:
    n = len(coords)
    d = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = cdist(coords[i], coords[j]).min()
    return d


def ca_distance(ca_coords: list[np.ndarray]) -> np.ndarray:
    xyz = np.stack(ca_coords, axis=0)
    return cdist(xyz, xyz).astype(np.float32)


def geodesic_distance(d: np.ndarray, radius: float) -> np.ndarray:
    mask = (d <= radius) & (d > 0) & np.isfinite(d)
    graph = np.where(mask, d, 0.0).astype(np.float64)
    return shortest_path(graph, method="D", directed=False, unweighted=False)


def spectral_embedding(d: np.ndarray, radius: float, n_components: int = 8) -> np.ndarray:
    mask = (d <= radius) & (d > 0) & np.isfinite(d)
    sigma = radius / 2.0
    w = np.where(mask, np.exp(-(d ** 2) / (2 * sigma ** 2)), 0.0)
    np.fill_diagonal(w, 0.0)
    L, _ = laplacian(w, normed=True, return_diag=True)
    _, vecs = np.linalg.eigh(L)
    k = min(n_components, vecs.shape[1] - 1)
    return vecs[:, 1:1 + k]


def adjacency(d: np.ndarray, cutoff: float) -> np.ndarray:
    a = (d <= cutoff) & (d > 0) & np.isfinite(d)
    return a.astype(np.uint8)


def build_all_settings(d: np.ndarray, tag: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for tau in (3.0, 5.0):
        out[f"{tag}_euclid_{int(tau)}A"] = adjacency(d, tau)
        out[f"{tag}_geodesic_{int(tau)}A_cut{int(tau)}A"] = adjacency(
            geodesic_distance(d, tau), tau)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("protein", type=Path, help="path to protein.pdb")
    ap.add_argument("--out", type=Path, default=Path.cwd() / "adjacency")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.protein.stem

    heavy, ca_coords, labels = parse_residues(args.protein)
    n = len(labels)
    print(f"{n} residues parsed from {args.protein.name}")
    missing_ca = sum(int(np.isnan(c).any()) for c in ca_coords)
    if missing_ca:
        print(f"  warning: {missing_ca} residues missing CA")

    d_mhad = mhad_distance(heavy)
    d_ca = ca_distance(ca_coords)

    settings: dict[str, np.ndarray] = {}
    settings.update(build_all_settings(d_mhad, "mhad"))
    settings.update(build_all_settings(d_ca, "ca"))

    n_pairs = n * (n - 1) / 2
    for name, a in settings.items():
        out = args.out / f"{stem}_{name}.npy"
        np.save(out, a)
        edges = int(a.sum()) // 2
        seq = sum(int(a[i, i + 1]) for i in range(n - 1))
        density = edges / n_pairs if n > 1 else 0.0
        print(f"  {name:30s} edges={edges:6d}  density={density:.4f}  "
              f"seq={seq:3d}  nonseq={edges - seq}")

    np.save(args.out / f"{stem}_labels.npy", np.array(labels))
    np.save(args.out / f"{stem}_spectral_embedding_mhad_3A.npy",
            spectral_embedding(d_mhad, 3.0))
    np.save(args.out / f"{stem}_spectral_embedding_mhad_5A.npy",
            spectral_embedding(d_mhad, 5.0))
    np.save(args.out / f"{stem}_spectral_embedding_ca_5A.npy",
            spectral_embedding(d_ca, 5.0))


if __name__ == "__main__":
    main()
