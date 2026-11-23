# runs_cc — scripts and outputs behind realqm_molecular_forces.tex (2026-11-20 … 11-23)

The browser logs (`*.log`) are not kept. Dumps were written to a session scratch directory and are gone; the `*.out` files are
the parsed results the article's source comments cite.

- **Runners.** `run.sh` drives one h2_beta.html run in headless Chrome (needs the page server on 8765 and the dump server on
  8767). `run_atom.sh` drives atom_simulator.html. `forces.py` reads the radial force on H out of a log.
- **Bond-length table.** `queue_geom1.sh` → `geom1.out` (observed geometry). `queue_scan.sh` → `scan.out` (R−0.1, R−0.2).
  `hf_bonds.py` gives the HF column kept in a source comment.
- **Dipoles.** `queue_dip.sh` → `dip.out`. Free-wall OH/HF (axisym): `xh_freecone.py`, `xh_split.py`, `xh_q.py`, with
  `xh_freecone.out` and `xh_split.out`.
- **Protein-surface polar groups.** `queue_frag.sh` → `frag.out`; `queue_fa.sh` → `fa.out` (formamide). `frag_ref.py` is the
  B3LYP reference.
- **C–C collapse.**
  - `c2_allelec.py` and `c2_cone.py`, axisym;
  - `c2_freecone.py` → `freecone_h005.out`, `c2ae_h005.out`;
  - `queue_c2bare.sh` → `c2bare.out`, `queue_c2h4hom.sh` → `c2h4hom.out`, `queue_h4.sh` → `h4.out`.
- **Vibrations.** `queue_freq.sh` → `freq.out`.
- **H2 switch figure.** `h2_switch.png`; `pauli_tnad.py` is the borrowed Thomas–Fermi test, not adopted.
- **Analysis layer.** `partition_dft.py` (H2O) and `partition_gen.py` (XH_n): B3LYP density partitioned by RealQM's wall rule.
- **He2 (not in the article).**
  - `he2_split.py`, `he2_cal.py`: split and shifted split; the cone option gives unphysical energies at h = 0.1;
  - `he2_dir.py`: 2hom with a free or Dirichlet wall; `he2_ref.py`: RHF and CCSD(T);
  - `queue_he2free.sh`, `queue_he2wedge.sh`: 3D;
  - outputs `he2_*.out`, `he2free.out`, `he2wedge.out`, figure `he2_split_geom.png`.
