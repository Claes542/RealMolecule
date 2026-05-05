# RealQM Reduced Model Database

A small library of per-protein reduced-model JSON records, generated from RealQM runs and used as input parameters for cell-scale Brownian dynamics.

## Schema

Each `.json` file describes one protein species:

| Field | Type | Description |
|---|---|---|
| `species` | string | informal name (e.g., "chignolin") |
| `pdb_id` | string | source PDB identifier |
| `sequence` | string | one-letter amino acid sequence |
| `n_residues` | int | residue count |
| `n_atoms_heavy` | int | heavy-atom count (excluding hydrogens) |
| `energy_Ha` | number / null | converged RealQM total energy, in atomic units |
| `net_charge_e` | int | net charge at neutral pH (Asp/Glu = -1, Lys/Arg = +1) |
| `mass_Da` | int | mass in daltons (~110 × n_residues) |
| `end_to_end_au` | number | Cα(N-term) - Cα(C-term) distance, atomic units |
| `end_to_end_AA` | number | same in Ångstrom |
| `bounding_box_au` | object | {x, y, z} ranges in atomic units |
| `hydrodynamic_radius_au` | number | half the maximum bounding-box dimension |
| `hydrophobicity_avg_kd` | number | sequence average on Kyte-Doolittle scale |
| `residues` | array | one entry per residue with chain/num/name/aa/hydrophobicity/sidechain_charge |
| `modes` | string / array | "TODO" or a list of {ω_cm, eigenvector} once Cα Hessian is computed |
| `surface_electrostatic` | string / object | "TODO" or {mesh, per-vertex potential, per-vertex hydrophobicity} |
| `provenance` | object | method, structure_source, extraction_tool, date |

## How to consume

The simplest downstream consumer is a Brownian-dynamics simulator that reads the JSON and uses the parameters to drive a population of copies. See `chignolin_bd.html` and `chignolin_ligand_bd.html` in the parent directory for working examples.

## How to contribute

1. Open `protein_reduce.html` in the parent directory.
2. Enter the PDB ID (e.g., `1L2Y`).
3. Click "Fetch & run" — wait for energy convergence.
4. Click "Extract now", then "Download JSON".
5. Place the downloaded file in this directory as `[PDB_ID].json`.
6. Update the index in `reduced_models.html`.

## Current entries

| PDB ID | Species | Residues | Status |
|---|---|---|---|
| 1UAO | chignolin (β-hairpin) | 10 | partial — sequence-derived parameters; modes & surface map TODO |

More entries planned: 1L2Y (Trp-cage), 1VII (villin headpiece), 1UBQ (ubiquitin), 2OOB (BPTI).

## Limitations

This is a research-grade database, not a curated production resource:
- Energy values may be from runs of varying convergence quality.
- Modes are computed on Cα coordinates only.
- Hydrophobicity uses the simple Kyte-Doolittle scale, not residue-context-dependent hydration.
- Surface-electrostatic maps are not yet implemented; the field is reserved for future entries.
