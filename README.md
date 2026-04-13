# RealQM — Ab Initio Molecular Dynamics

RealQM is based on a new form of Schrodinger's equation which has the form of classical multi-phase continuum mechanics in 3D for a system of non-overlapping electron densities interacting by Coulomb potentials. Computational complexity scales with the number of mesh points in 3D and so offers laptop computational simulation of large molecular systems including ab initio protein folding. Runs entirely in the browser using WebGPU compute shaders on any laptop GPU.

**[Go to Demo Gallery](https://claes542.github.io/RealMolecule/gallery.html)** (requires Chrome 113+ or Safari 17+ with WebGPU)

## Quick start

1. Clone and serve locally:
   ```bash
   git clone https://github.com/Claes542/RealMolecule.git
   cd RealMolecule
   python3 -m http.server 8000
   ```
2. Open `http://localhost:8000/gallery.html` in Chrome or Safari

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

#### Hairpin folding from quantum H-bonds

> **12-residue polyglycine beta-hairpin** folds from 150° to ~105° opening angle driven purely by quantum mechanical forces — interstrand N-H···O=C hydrogen bonds pull the two strands together. No empirical force field is used. The folding stalls at ~105° where H-bond attraction balances backbone repulsion. Adding explicit water does not help: the hydrophobic effect that completes folding in real biology is entropic, which our electronic energy solver cannot capture. [Run dry →](hairpin_bent_dry.html) | [Side by side →](hairpin_sidebyside.html)

#### Alpha-helix formation from near-linear chain

https://github.com/Claes542/H2O/raw/main/helix_formation.webm

200³ grid version (faster, 2D density view): https://github.com/Claes542/H2O/raw/main/helix_formation_200.webm

> **8-residue polyglycine** starts as a nearly straight chain (radius 0.5 Å, rise 3.0 Å/residue) and folds into an alpha helix — driven purely by ab initio quantum forces. No empirical force fields are used.

| Parameter | Start | Final | Ideal α-helix |
|-----------|-------|-------|---------------|
| Helix score | ~5% | **67%** | 100% |
| Radius | 0.5 Å | **2.1 Å** | 2.3 Å |
| Rise/residue | 3.0 Å | **1.63 Å** | 1.5 Å |
| i→i+4 H-bonds | 0/4 | **2/4** (5 Å) | 4/4 (1.9 Å) |

The chain compresses axially and curls outward into near-ideal helical geometry. Two i→i+4 hydrogen bonds form spontaneously: carbonyl oxygen of residue i attracted to amide hydrogen of residue i+4 — the defining feature of the alpha helix. Rise converges to within 9% of ideal, radius reaches 91% of ideal. [Run it yourself →](alpha_helix_dry_300.html)

**Method**: Per-residue rigid-body translation by net quantum forces on a 300³ grid (45 au box), with Ca-Ca SHAKE constraints (±15% of 3.8 Å). Mild radial helical bias (0.03) assists curling. Electronic structure re-solved from scratch after each geometry update.

#### Alpha-helix formation — solvated

https://github.com/Claes542/H2O/raw/main/helix_solvated.webm

> Same 8-residue polyglycine helix formation, now surrounded by a single shell of explicit water molecules (~150 total atoms on 300³ grid). [Run it yourself →](alpha_helix_solvated_dyn.html)

#### Polyglycine vs chignolin: the role of side chains

| | **Polyglycine** (no side chains) | **Chignolin** GYDPETGTWG (real side chains) |
|---|---|---|
| **Start** | 150° | 135° |
| **Result** | → 105° (**folds**) | → unfolds |
| **Why** | Backbone H-bonds dominate | Side-chain repulsion overwhelms H-bonds |

[Run side by side →](hairpin_vs_chignolin.html)

#### What this tells us about water and folding

Our solver computes electronic energy (Coulomb, kinetic, exchange) but not entropy. Testing shows:

1. **Water is not essential for H-bond driven folding** — polyglycine folds to ~105° in vacuum. Adding explicit water does not improve folding; both dry and solvated converge to the same angle.
2. **Real proteins with side chains need more than H-bonds** — chignolin unfolds in vacuum because side-chain electron repulsion overwhelms backbone attraction. In real biology, the hydrophobic effect (entropic — water molecules lose orientational freedom near hydrophobic surfaces) overcomes this repulsion and drives compact packing. Our solver cannot capture this entropic force.
3. **SASA implicit solvent solves this** — a single surface-tension parameter (γ=5.0) replaces all hand-tuned native contacts. Exposed residues feel an inward force toward the protein centroid, proportional to their solvent-accessible surface area. No explicit water molecules needed.

#### Near-blind folding: two generic rules

The closest to blind folding we achieve uses **no knowledge of the native 3D structure**:

1. **Universal i→i+4 H-bond bias** — applied to every residue in the chain. Helices form where quantum forces support them, stay open elsewhere. No need to specify which segments are helical.
2. **SASA hydrophobic pressure** — one parameter (γ=5.0) drives compaction. No native contacts.

Tested on **Villin HP35** (35 residues, 3-helix bundle): 7/9 helix H-bonds form at 2.4–3.2 Å, Phe core packs to 5.6 Å (target 6.0), with zero knowledge of which residues are helical or where the hydrophobic core is.

| Protein | Residues | H-bond biases | Packing | Result |
|---------|----------|--------------|---------|--------|
| Polyglycine | 12 | None (quantum only) | None | Folds 150° → 105° |
| Chignolin | 10 | None | SASA | Folds 135° → 40° |
| Trp-cage | 20 | Helix i→i+4 | SASA | Helix + core form |
| BBA5 | 23 | Helix + sheet | SASA | 3/5 H-bonds, core packs |
| Villin | 35 | Universal i→i+4 | SASA | 7/9 H-bonds, core at target |
| Crambin | 46 | Helix + sheet | SASA | 3/5 H-bonds, disulfides approach |

#### How this differs from existing approaches

Protein folding methods typically fall into two categories: (1) classical MD with empirical force fields (AMBER, CHARMM — thousands of fitted parameters) plus implicit or explicit solvent, or (2) ML-based prediction (AlphaFold, ESMFold — no physics). Quantum mechanical methods (FMO, DFTB) have been applied to protein energetics but not to folding dynamics due to computational cost.

This solver occupies a gap: **ab initio quantum forces + one implicit solvent parameter → folding**. The electronic structure is solved from the Schrödinger equation on a real-space grid — no empirical force field, no fitted Lennard-Jones or torsion parameters. SASA adds a single surface-tension coefficient (γ) to approximate the hydrophobic effect. Universal i→i+4 H-bond biases require no knowledge of the native structure.

**Caveats**: The H-bond biases are empirical steering forces (not fully ab initio). Grid resolution limits quantitative accuracy. Folding rates and free energies have not been validated against experiment. Beta-sheet formation still requires specifying strand pairings. This is a proof of concept, not a production tool.

#### Validation suite

| Test | Atoms | Result |
|------|-------|--------|
| [Alanine dipeptide](ala_dipeptide.html) | 22 (3D) | Bond lengths match PySCF HF reference for pseudopotentials |
| [Formamide dimer](formamide_dimer.html) | 12 (3D) | N-H···O=C hydrogen bond geometry correct (H···O ~1.9 Å) |
| [Alpha-helix stability](alpha_helix.html) | 35 (3D) | 5-residue polyglycine helix stable under quantum dynamics |
| [Alpha-helix formation](alpha_helix_dry_300.html) | ~50 (3D) | 8-residue near-linear → 67% helix, 2/4 H-bonds ([video](helix_formation.webm)) |
| [Hairpin folding (dry)](hairpin_bent_dry.html) | 87 (2D) | Folds from 150° → 105° via quantum H-bonds; water does not improve |
| [Chignolin folding](chignolin_dry.html) | ~90 (3D) | GYDPETGTWG folds 135° → 40° with SASA hydrophobic pressure ([video](chignolin_fold.webm)) |
| [Trp-cage folding](trpcage_fold.html) | ~200 (3D) | H-bond biases + SASA, no native contacts |
| [Villin folding](villin_fold.html) | ~350 (3D) | Universal i→i+4 + SASA, 7/9 H-bonds, blind fold |
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
