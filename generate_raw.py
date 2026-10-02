"""Generate independent synthetic pooled binding panels.

This is an idealized equilibrium simulator, not measured chemical data.
Copyright 2026 Iqbalez. Licensed under the MIT License.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

VERSION = "1.0.0"
LIGANDS = 10
FEATURES = 4
ASSAYS = 2
MOLECULES_PER_ASSAY = 200
BETA = np.array([0.8, -0.6, 0.45, -0.25], dtype=float)


def panel(master_seed: str, index: int) -> dict:
    digest = hashlib.sha256(f"{master_seed}:panel:{index}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:16], "big"))
    descriptors = rng.normal(size=(LIGANDS, FEATURES))
    log_affinity = (
        descriptors @ BETA
        + 0.32 * np.sin(1.7 * descriptors[:, 0] * descriptors[:, 1])
        + rng.normal(0, 1.3, LIGANDS)
    )
    affinity = np.exp(log_affinity)
    assay_conc = rng.lognormal(-2.0, 0.6, (ASSAYS, LIGANDS))
    assay_conc *= rng.random((ASSAYS, LIGANDS)) < 0.52
    for row in assay_conc:
        if np.count_nonzero(row) < 2:
            selected = rng.choice(LIGANDS, 3, replace=False)
            row[selected] = rng.lognormal(-2.0, 0.6, 3)
    query_conc = rng.lognormal(-0.1, 0.85, LIGANDS)
    z = assay_conc @ affinity
    occupancy = z / (1.0 + z)
    bound_counts = rng.binomial(MOLECULES_PER_ASSAY, occupancy)
    winner = int(np.argmax(query_conc * affinity))
    return {
        "panel_key": hashlib.sha256(f"{master_seed}:id:{index}".encode()).hexdigest()[:24],
        "descriptors": np.round(descriptors, 8).tolist(),
        "assay_concentrations": np.round(assay_conc, 8).tolist(),
        "bound_counts": [int(x) for x in bound_counts],
        "query_concentrations": np.round(query_conc, 8).tolist(),
        "winner_index": winner,
    }


def generate(seed: str, count: int, out: Path):
    if len(seed) < 32:
        raise ValueError("Use a private master seed of at least 32 characters")
    if count < 100:
        raise ValueError("At least 100 independent panels are required")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "panels.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for index in range(count):
            f.write(json.dumps(panel(seed, index), separators=(",", ":"), allow_nan=False) + "\n")
    metadata = {
        "dataset_title": "Synthetic Pooled Binding Assay Panels",
        "generator_version": VERSION,
        "panels": count,
        "ligands_per_panel": LIGANDS,
        "assays_per_panel": ASSAYS,
        "molecules_per_assay": MOLECULES_PER_ASSAY,
        "seed_sha256": hashlib.sha256(seed.encode()).hexdigest(),
        "synthetic": True,
    }
    (out / "GENERATION_METADATA.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate(args.seed, args.count, args.out)
