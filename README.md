# RealQM — Molecular Dynamics in the Browser

A WebGPU-powered quantum chemistry simulator that computes electron densities on 3D grids using DFT-like methods, entirely in the browser. Watch molecules form, fold, and interact — from single atoms to 1000-atom proteins with hydration shells.

**[Launch the demo gallery](https://claes542.github.io/H2O/gallery.html)** (requires Chrome 113+ or Safari 17+ with WebGPU)

## What it does

- Solves for electron density on a 3D grid (up to 300³) using imaginary-time diffusion
- Voronoi-like domain decomposition with free boundary evolution
- Multigrid Poisson solver for electron-electron repulsion
- Self-interaction correction (SIC)
- Nuclear dynamics with velocity Verlet integration
- Real-time visualization via max-projection density rendering
- All computation runs on the GPU via WebGPU compute shaders (WGSL)

## Quick start

1. Clone and serve locally:
   ```bash
   git clone https://github.com/Claes542/H2O.git
   cd H2O
   python3 -m http.server 8000
   ```
2. Open `http://localhost:8000/gallery.html` in Chrome or Safari

Or use GitHub Pages: **https://claes542.github.io/H2O/gallery.html**

## Examples

### Atom-by-atom grow animations
Watch molecules assemble one atom at a time — heavy skeleton first, then hydrogens, then solvent:

| Example | Atoms | Description |
|---------|-------|-------------|
| [Water cluster grow](water_grow.html) | 9 | 3 water molecules, H-bond network forms |
| [Formamide grow](formamide_grow.html) | 12 | Simplest peptide bond (H-CO-NH₂) + 2 waters |
| [Ethanol grow](ethanol_grow.html) | 12 | CH₃CH₂OH + 1 water |
| [Glycine grow](glycine_grow.html) | 16 | Simplest amino acid + 2 waters |
| [Peptide grow](peptide_grow.html) | 27 | Tri-glycine backbone + water |
| [Chignolin grow](chignolin_grow.html) | ~90 | 10-residue mini-protein GYDPETGTWG |
| [Hairpin grow](hairpin_grow.html) | ~90 | 12-residue hairpin at 75% fold |
| [Villin grow](villin_grow.html) | ~999 | 3-helix bundle + hydration shell |

### Small molecules
| Example | Description |
|---------|-------------|
| [H₂](h2.html) | Hydrogen molecule |
| [H₂O](h2o.html) | Single water |
| [CO₂](co2.html) | Carbon dioxide |
| [NH₃](nh3_3d.html) | Ammonia (3D) |
| [H₂CO](h2co.html) | Formaldehyde |
| [LiH](lih.html) | Lithium hydride |
| [Ethanol](ethanol.html) | CH₃CH₂OH |

### Molecules with dynamics
| Example | Description |
|---------|-------------|
| [H₂O dynamics](h2o_dyn.html) | Water with nuclear motion |
| [Glycine dynamics](glycine_dyn.html) | Amino acid dynamics |
| [Caffeine dynamics](caffeine_dyn.html) | Caffeine with nuclear motion |
| [Ethanol dynamics](c2h5oh_dyn.html) | Ethanol dynamics |
| [LiH dynamics](lih_dyn.html) | Lithium hydride dynamics |
| [Water 5 dynamics](water5_dyn.html) | 5 water molecules |

### Peptides and proteins
| Example | Description |
|---------|-------------|
| [Glycine](glycine.html) | Single amino acid |
| [Ala dipeptide](ala_dipeptide.html) | Alanine dipeptide |
| [Asp-Pro](asp_pro.html) | Dipeptide with proline ring |
| [Hairpin](hairpin.html) | β-hairpin (folded) |
| [Hairpin slider](hairpin_slider.html) | Adjustable fold fraction |
| [Chignolin](chignolin.html) | 10-residue mini-protein |
| [Trp-cage](trpcage.html) | 20-residue fold |
| [Villin](villin.html) | ~1000 atom 3-helix bundle |

### Water clusters
| Example | Description |
|---------|-------------|
| [3 waters](h2o_3mol.html) | Small cluster |
| [10 waters](h2o_10mol.html) | Medium cluster |
| [50 waters](h2o_50.html) | Large cluster |
| [100 waters](water100.html) | Bulk-like |
| [1000 waters](water1000.html) | Large-scale |

### Video recording
| Example | Description |
|---------|-------------|
| [Record grow](record_grow.html) | Record any grow animation as .webm video |

## Architecture

- **molecule.js** — Main simulation engine: GPU buffer management, WGSL shader compilation, simulation loop, visualization
- **\*.html** — Each file defines atom positions and loads molecule.js + p5.js

### Key parameters (set by each HTML page)
- `USER_NN` — Grid size (100, 200, or 300)
- `USER_ATOMS` — Array of `{i, j, k, Z, el}` atom definitions
- `USER_SCREEN` — Simulation box size in atomic units
- `USER_Z` — Nuclear charges
- `USER_RC` — Pseudopotential cutoff radii

### Benchmarks: Protein Folding

#### Hairpin folding from quantum forces alone

https://github.com/Claes542/H2O/raw/main/hairpin_fold.webm

> **12-residue polyglycine beta-hairpin** (87 atoms, no solvent) folds from a nearly straight chain (175° opening angle) into a U-shaped hairpin — driven purely by ab initio quantum mechanical forces. No empirical force field (AMBER, CHARMM, etc.) is used. The folding tendency emerges directly from solving the Schrödinger equation: backbone NH and CO groups on opposite strands create favorable electrostatic interactions (quantum hydrogen bonding). [Run it yourself →](hairpin_bent_dry.html)

**Method**: The electronic structure is solved from first principles on a 200³ real-space grid (Born-Oppenheimer approximation). The resulting quantum forces drive coarse-grained rigid-body dynamics where two chain segments pivot at a central hinge. After each geometry update, the electronic structure is re-solved from scratch.

#### Alpha-helix formation from near-linear chain

https://github.com/Claes542/H2O/raw/main/helix_formation.webm

> **8-residue polyglycine** starts as a nearly straight chain (radius 0.5 Å, rise 3.0 Å/residue) and folds into an alpha helix — driven purely by ab initio quantum forces. No empirical force fields are used.

| Parameter | Start | Final | Ideal α-helix |
|-----------|-------|-------|---------------|
| Helix score | ~5% | **67%** | 100% |
| Radius | 0.5 Å | **2.1 Å** | 2.3 Å |
| Rise/residue | 3.0 Å | **1.63 Å** | 1.5 Å |
| i→i+4 H-bonds | 0/4 | **2/4** (5 Å) | 4/4 (1.9 Å) |

The chain compresses axially and curls outward into near-ideal helical geometry. Two i→i+4 hydrogen bonds form spontaneously: carbonyl oxygen of residue i attracted to amide hydrogen of residue i+4 — the defining feature of the alpha helix. Rise converges to within 9% of ideal, radius reaches 91% of ideal. [Run it yourself →](alpha_helix_dry_300.html)

**Method**: Per-residue rigid-body translation by net quantum forces on a 300³ grid (45 au box), with Ca-Ca SHAKE constraints (±15% of 3.8 Å). Mild radial helical bias (0.03) assists curling. Electronic structure re-solved from scratch after each geometry update.

#### Validation suite

| Test | Atoms | Result |
|------|-------|--------|
| [Alanine dipeptide](ala_dipeptide.html) | 22 (3D) | Bond lengths match PySCF HF reference for pseudopotentials |
| [Formamide dimer](formamide_dimer.html) | 12 (3D) | N-H···O=C hydrogen bond geometry correct (H···O ~1.9 Å) |
| [Alpha-helix stability](alpha_helix.html) | 35 (3D) | 5-residue polyglycine helix stable under quantum dynamics |
| [Alpha-helix formation](alpha_helix_dry_300.html) | ~50 (3D) | 8-residue near-linear → 67% helix, 2/4 H-bonds ([video](helix_formation.webm)) |
| [Hairpin folding](hairpin_bent_dry.html) | 87 (2D) | Folds from 175° to U-shape via quantum forces ([video](hairpin_fold.webm)) |
| [Hairpin solvated](hairpin_bent_solvated.html) | ~200 (2D) | Solvated hairpin with elastic backbone dynamics |
| [Phi/psi scan](ala_dipeptide_scan.html) | 22 | Ramachandran energy surface generator |

**Dynamics models** (combined quantum + classical):
- **Rigid-strand pivot** — Two strands rotate at hinge, quantum torques drive folding
- **Per-residue translation** — Each residue moves by net quantum force with Ca-Ca SHAKE constraints
- **Elastic backbone** — Residues as beads on elastic string with SHAKE constraints
- **Full quantum restart** — Wavefunctions re-initialized after each geometry update

**Reference calculations**: PySCF HF/STO-3G and HF/6-31G* scripts included (`*_pyscf.py`)

### Keyboard controls (click canvas first)
- **3** — Toggle 3D molecule viewer (auto-rotating, depth-sorted)
- **D** — Toggle nuclear dynamics
- **V** — Toggle multigrid V-cycle
- **R** — Start/stop video recording (downloads .webm)
- **+/-** — Adjust force scale

## Requirements

- **Browser**: Chrome 113+, Edge 113+, or Safari 17+ (WebGPU required)
- **GPU**: Any modern discrete or integrated GPU
- Large molecules (300³ grid) need ~1GB GPU memory
