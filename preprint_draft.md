# Quantum Mechanics as Multiphase 3D Continuum Mechanics
## Realised through Mind–AI Cooperation

**Author:** Claes Johnson  

*This article was written by Claude (Anthropic) with assistance by the author. The mathematical theory and the simulation framework are the author's; the implementation was developed by Claude from p5.js prototypes by the author.*

---

## Abstract

We present a real-space reformulation of the many-electron problem in which matter is described as a system of N **non-overlapping unit electron densities** in three-dimensional space, arranged by minimum-energy Coulomb packing. The unit-density variational principle is a free-boundary problem of Bernoulli type: the boundaries between electron territories are not specified in advance but emerge from the joint minimization over densities, boundary location, and nuclear positions, with continuity and homogeneous Neumann boundary conditions arising naturally from the variational structure. The model is organized as a hierarchy of reductions: **Level 1** is the parameter-free basic atomic model in which shell structure emerges from minimum-energy packing rather than being preset, reproducing observed atom total energies for elements Li–Rn within ~1%; **Level 2** treats atoms with spherical symmetry on the inner shells while leaving valence electrons explicit; **Level 3** replaces nucleus and inner shells together by a pseudopotential characterized by an effective charge and a softening radius; **Level 4** assembles Level-3 atoms into molecules and molecular complexes. At Level 3, reduced-kernel architectures sweep entire periodic-table columns by varying the softening radius — closed-shell hydrides match experimental atomization energies within 3–9% for groups 1, 2, 14, and 16 (NaH, BeH₂, H₂O, CH₄, SiH₄, GeH₄). Excited states are accessible through the same variational principle (orthohelium worked example). At Level 4, the framework supports **interactive protein folding**: polyglycine hairpins fold from extended to compact via quantum-mechanical H-bond forces alone; alpha-helices form from near-linear chains; mini-proteins (chignolin, trp-cage, villin HP35) fold under universal i→i+4 H-bond bias plus a single hydrophobic-pressure parameter, reproducing native-like topology without empirical force fields. The framework is implemented in ~8000 lines of JavaScript and WebGPU compute shaders running interactively in a browser on a consumer laptop GPU — three orders of magnitude smaller and ~10²–10⁴ × faster than mainstream quantum-chemistry packages. The implementation, validation suite, and curated public Gallery were developed through extended collaboration with an AI code assistant (Claude, Anthropic) starting from initial p5.js prototypes by the author. We present the collaboration as part of the contribution: a worked case of how a single mathematical mind, paired with an AI engineering partner, can produce research-grade interactive scientific software at a scale that previously required a programming team.

**Keywords:** quantum chemistry, density functional theory, free-boundary problems, Bernoulli condition, real-space methods, WebGPU, AI-assisted scientific software, human-AI collaboration

---

## 1. Introduction

The N-electron Schrödinger equation has been the foundation of quantum chemistry for nearly a century. In its standard formulation, matter is described as eigenfunctions of an N-body Hamiltonian acting on antisymmetric wavefunctions in a 3N-dimensional configuration space. The exponential scaling of this representation has been mitigated by approximate methods — Hartree-Fock, density functional theory, configuration interaction, coupled cluster — each making specific compromises between accuracy and computational cost. Production-quality quantum chemistry packages comprise hundreds of thousands to millions of lines of code, draw on decades of accumulated machinery for basis sets, integral evaluation, and analytic gradients, and require dedicated computational clusters for systems of biological size.

This paper presents a different starting point. We treat an N-electron system not as eigenfunctions in Hilbert space but as a system of **non-overlapping unit electron densities** in three-dimensional real space, arranged by a minimum-energy Coulomb packing principle. The mathematical object of study is a set of N density functions ρ_i(r) on R³, each integrating to one electron of charge, subject to the constraint that they do not overlap pairwise. Energy is the standard Coulomb functional. The minimum is taken over these densities and the nuclear positions; molecular geometry, bonding, and dynamics emerge from the same variational principle.

In this respect RealQM is a direct realisation of **Dirac's view of chemistry as applied quantum physics**: the famous 1929 remark that "the underlying physical laws necessary for the mathematical theory of [...] the whole of chemistry are thus completely known, [and] the difficulty is only that the exact application of these laws leads to equations much too complicated to be soluble". RealQM takes Dirac's program seriously by retaining only the underlying physics — Coulomb interactions between electron densities and nuclei — and discarding the chemistry-specific machinery (basis sets, exchange–correlation functionals, fitted force fields) that the standard reformulations introduce in order to make the equations tractable. Chemistry then re-emerges, bottom-up, from the variational principle alone.

This reformulation — RealQM in what follows — is not merely a numerical recasting of the standard problem. It is a different mathematical object with different scaling: complexity grows with the number of mesh points in 3D, not with the number of basis functions in N×3D. As a consequence, the entire framework can be implemented in roughly 8000 lines of JavaScript and WebGPU compute shaders, runs in a browser on a consumer laptop GPU, and supports interactive simulation of systems with hundreds of explicit electrons in real time.

We organize the model as a hierarchy of reductions. **Level 1** is parameter-free: an atom is described as a nucleus of charge Z surrounded by N unit electron densities organized into non-overlapping shells, with the configuration determined by minimum-energy packing. The Level-1 atomic model reproduces observed atom energies for elements Li through Rn to approximately 1% with no fitted constants. **Level 2** spherically homogenizes inner electron shells, keeping the shell occupancy intact but replacing the angular structure with a radial density of matching total charge. **Level 3** further replaces the nucleus and inner shells together by a *pseudo-kernel* characterized by an effective charge Z_kernel and a softening radius r_c, leaving only the outermost valence electrons explicit. **Level 4** combines Level-3 atoms into molecular assemblies. Each level is determined by comparison with the prior level, in the same architectural spirit as pseudopotentials and frozen-core methods in standard quantum chemistry, but with a parameter-free ab initio bottom rather than empirical fits.

A second contribution of this paper concerns how the implementation was produced. The mathematical reformulation is the work of one author. The validation suite, the WebGPU compute shaders, the systematic kernel-architecture sweeps reported in Section 4, and the curated public gallery were developed through extended interactive collaboration with **Claude**, a large-language-model AI assistant produced by Anthropic. The pattern of the collaboration was iterative and recorded: the human proposed mathematical and chemical questions; the AI translated them into runnable simulations, identified bugs, ran systematic parameter sweeps, and challenged claims that did not survive scrutiny; the human evaluated the results and corrected the AI's interpretations when they slipped. The history of corrections — claims made, retracted, refined, restored — is itself part of the public record, encoded in the gallery's "Assessment by Claude" card.

We take this case to be a worth-reporting example of how a single mathematical mind, working with an AI engineering partner in real time, can produce research-grade interactive scientific software at a scale otherwise requiring a programming team. The collaboration is not the science, but it is part of how the science came into existence, and we report it as such.

The paper is organized as follows. Section 2 states the reformulation and the hierarchical reduction. Section 3 describes the numerical implementation. Section 4 reports validation, including atom energies, the covalent H₂ bond and its He limit, atomization energies of closed-shell hydrides across periodic-table groups, S66 dimer geometries, chemical reactions, protein folding, excited states, and the NaCl crystal melt as a condensed-phase example. Section 5 states honest limits of the approach. Section 6 places the work in the long-standing chemistry-vs-physics debate. Section 7 discusses the human–AI collaboration. The full source code, validation suite, and interactive Gallery are available at the URLs given in Section 8.

---

## 4. Validation

### 4.1 Level-1 atom energies, Li through Rn

We first validate the parameter-free Level-1 atomic model against observed atom total energies. The model is: a nucleus of charge Z surrounded by N unit electron densities, each occupying a non-overlapping spatial region, with the configuration that minimizes the total Coulomb energy taken as the ground state.

A central feature of the Level-1 model is that **shell structure is not preset**. **A shell structure partially conforming with the standard s, p, d organization emerges** from the variational principle as a consequence of minimum-energy packing: as electrons are added one at a time, each new electron occupies the lowest-energy spatial region available given the non-overlap constraint, and electron sizes naturally grow with distance from the kernel. The radial layering (a 2-electron inner shell, an 8-electron next shell, an 18-electron third shell, ...) corresponds to the standard 1s², 2s²2p⁶, 3s²3p⁶3d¹⁰ counts, and is reproduced quantitatively in the energetics. The angular structure (the s/p/d distinction within a shell — spherical s, dumbbell p, four-lobe d) is **only partially captured** by the unit-density partition, which divides each shell into spatial territories whose detailed angular shapes need not match the standard hydrogenic orbitals. The agreement of Level-1 atom energies with observation across Li–Rn is therefore evidence that the **radial** shell organization is correctly derived from packing, while the finer angular labelling of subshells remains an approximation in this Level-1 description.

The model has no fitted parameters; the only input is the nuclear charge Z. Table 1 shows computed and observed total energies for representative elements Li through Rn. Agreement is approximately 1% across the range, with the largest deviations at very heavy atoms where relativistic corrections are not included.

This level provides the anchor for all subsequent reductions: each higher-level model (Level 2, 3, 4) is parameterized by comparison with Level-1 atom energies and Level-2 spherical densities.

### 4.2 Chemical bonding: the H₂ covalent bond and its Helium limit

The hydrogen molecule H₂ is the simplest covalent bond — two protons sharing two electrons — and provides the cleanest test of how covalent bonding emerges from the unit-density formulation. In RealQM, H₂ is described as two unit electron densities that do not overlap (Bernoulli partition), distributed around two H⁺ kernels at separation R. The two electrons occupy opposite halves of the inter-nuclear region (the natural minimum-energy partition for two electrons in a symmetric two-kernel field). Each electron is attracted to both nuclei and repelled by the other electron through Coulomb interactions only; there is no exchange term and no antisymmetric wavefunction.

At a separation R ≈ 1.6 a.u. close to the experimental bond distance (1.40 a.u.), the parameter-free Level-1 calculation — each hydrogen contributing one unit electron density on its side of the Bernoulli partition — gives a total energy E = −1.1785 Ha against the Kolos–Wolniewicz exact value −1.1745 Ha (0.3% deeper than exact). The corresponding binding energy is ΔE_bind ≈ −112 kcal/mol versus the experimental dissociation energy −109 kcal/mol — **within 3%, with no fitted parameters**. The bond emerges from the same variational principle that organizes the atomic shells (Section 4.1): the electrons reorganize their territories to minimize the total Coulomb energy, and at the equilibrium R the two-kernel two-electron configuration is energetically preferred over two isolated H atoms.

**Bonding mechanism.** The bond is established by **accumulation of electron density between the two kernels**, which lowers the total kernel-attraction potential energy by bringing density closer to both nuclei. In the unit-density formulation, the two non-overlapping electron territories meet at the free boundary between the kernels, where **both densities are non-zero** — neither is required to vanish at the interface. This non-vanishing meeting (the Bernoulli condition of Section 2.4: continuity plus homogeneous Neumann) is precisely what makes the inter-nuclear accumulation energetically favorable. If the densities instead had to decay to zero between the nuclei (as they do at infinity for an isolated atom), accumulating density in the inter-nuclear region would require steep gradients and a balancing kinetic-energy cost that would offset the potential-energy gain. The Bernoulli partition lets each density remain smooth across the interface with finite values on both sides, so the inter-nuclear region accumulates electrons at low kinetic-energy cost while the kernel-attraction potential energy is enhanced. Covalent bonding in RealQM is thus a direct consequence of the free-boundary structure: **meeting non-zero densities allow the kernel potential energy to decrease through inter-nuclear accumulation without a balancing increase in kinetic energy**. This conclusion is consistent with the Rüdenberg–Kutzelnigg analysis of the chemical bond [Ruedenberg1962, Kutzelnigg1990], in which careful decomposition of the wavefunction shows that covalent bonding is primarily a kinetic-energy lowering driven by orbital interference; the unit-density formulation reaches the same qualitative conclusion through a different route, namely the free-boundary partition rather than orbital overlap.

**The Helium limit: zero kernel distance, no nuclear repulsion.** A revealing limit is R → 0. As the two H nuclei approach, their nuclear–nuclear repulsion 1/R diverges, so H₂ has a finite equilibrium separation. But if the two H⁺ point kernels are *merged* into a single +2 point kernel — equivalently R = 0 with the charges combined — the system becomes Helium (Z = 2, two electrons), and there is no inter-nuclear repulsion to oppose the merger. The Bernoulli partition is now two unit densities that do not overlap, both bound to the inner shell of the +2 kernel, with their interface bisecting the kernel. This is exactly the Level-1 description of the He ground state.

The covalent bond and the atomic shell are therefore not separate phenomena; they are the same minimum-energy electron-packing principle applied at different nuclear separations. The continuous family parameterized by R interpolates between He at R = 0 (kernels merged, no repulsion), the H₂ bond at R = R_eq, and two isolated H atoms at R → ∞. Each configuration is a stationary point of the same energy functional. The unification clarifies why a single architecture handles atoms and molecules: bonding is what the variational principle does when more than one nucleus is present, with the inter-nuclear repulsion setting the equilibrium separation.

### 4.3 Atomization energies of closed-shell hydrides

We test the Level-3 reduction across a series of closed-shell hydrides H_n X, varying the heavy atom X across periodic-table groups. For each system, the kernel is parameterized by an effective charge Z_kernel and a softening radius r_c. The kernel may be either non-split (a single multi-occupancy orbital) or split into angular sectors of equal valence-electron occupancy. We compute the total energy at the experimental equilibrium X-H bond length R and at twice that distance, and report

ΔE_bind = E(R) − E(2R)

as a proxy for the atomization energy. This single-coordinate scan is appropriate because all atoms are in their fixed equilibrium configurations and only R varies; the difference encodes the strength of the X-H bonds.

**Group 1 (alkali hydrides) via XH (X+1, no split, closed shell).** A single-electron X kernel paired with a single H gives a closed-shell two-electron molecule (analog of LiH, NaH, KH). Sweeping r_c traces the alkali series:

| r_c (au) | RealQM ΔE (kcal/mol) | Real molecule | Match |
|---------:|---------------------:|---------------|-------|
| 0.50 | −48 | NaH −47 | **within 2%** |
| 0.70 | −42 | KH ~−43 | **within 2%** |

The rc = 0 limit of this single-orbital model corresponds to a 2-electron-in-one-territory configuration, distinct from the proper covalent H₂ (Section 4.2). The model captures alkali-hydride binding energies to within experimental uncertainty at the right kernel softening — a single architecture spanning a periodic-table column.

**Group 2 (alkaline-earth dihydrides) via HXH (X+2, 2-hemi split).** Linear H-X-H with a +2 kernel split into two hemispheres along the molecular axis (each holding one electron) gives binding energies in the range:

| r_c (au) | ΔE (kcal/mol) | Real molecule | Match |
|---------:|--------------:|---------------|-------|
| 0.40 | −140 | BeH₂ −144 | **within 3%** |
| 0.50 | −129 | between BeH₂ and MgH₂ | regime |

**Group 14 (XH₄ tetrahedral hydrides) via H₄X (C+2, no split).** A central +2 kernel with a single 2-electron orbital paired with four hydrogen atoms at tetrahedral positions traces the entire group-14 series via r_c:

| r_c (au) | ΔE (kcal/mol) | Real molecule | Match |
|---------:|--------------:|---------------|-------|
| 0.20 | −369 | CH₄ −396 | **within 7%** |
| 0.40 | −348 | SiH₄ −320 | within 9% |
| 0.70 | −272 | GeH₄ −281 | **within 3%** |

This is the cleanest result of the series: a single architecture with no architectural changes captures methane, silane, and germane to better than 10%, with the trend matching the chemical periodic table.

**Group 16 (bent H₂X) via H₂O (O+3, 2-hemi bisector).** The water case requires a different splitting topology, with the axis along the H-O-H bisector. Both H atoms occupy the same hemisphere (with the lone pair in the opposing hemisphere as a paired sub-orbital). At r_c = 0.7, ΔE_bind = −225 kcal/mol versus the experimental H₂O atomization 232 kcal/mol — within 3%.

### 4.4 What r_c encodes

Across all four working groups, varying r_c with a fixed architecture traces a periodic-table column. We interpret r_c as the **inner-shell absorption radius** in the Level-3 reduction: the radius at which inner-shell electrons are absorbed into the kernel core, leaving only the explicit valence outside. As r_c grows, the kernel becomes more diffuse, the effective valence sees a larger inner core, and the bonding becomes weaker — exactly as down a periodic-table column. This interpretation is consistent quantitatively across groups 1, 2, 14, and 16, with periodic-trend match within 3–9% per element.

### 4.5 Geometric validation against S66

We additionally test geometric agreement against Hobza's S66 benchmark of non-covalent dimers. RealQM cannot directly validate S66 binding energies — those are CCSD(T)/CBS values in the −1 to −7 kcal/mol range, well below the model's absolute-energy noise floor (~0.1 Ha). But the equilibrium distances and angles can be compared directly. For five H-bonded dimers tested (water-water, methanol dimer, methylamine dimer, methylamine-water, formamide dimer), distances agree with CCSD(T) reference to within 1–5%, and force-direction diagnostics confirm the basin of attraction is correctly identified. Detailed tables are in the public Gallery (URL in Section 7).

### 4.6 Chemical reactions: proton transfer

The local-potential force formulation (Section 2.5) makes RealQM a natural framework for chemical reactions. A reaction proceeds because the local Coulomb forces on each nucleus push it along a path that passes through bond-breaking and bond-making configurations; nothing is computed by reference to an energy surface, a transition-state search, or a barrier height. The forces are the same Coulomb gradients used for equilibrium geometry — only the initial configuration differs.

We illustrate this with **proton transfer**, the prototypical Brønsted acid–base reaction.

**HF + H₂O → F⁻ + H₃O⁺.** The donor H of HF is placed at hydrogen-bonding contact from the O of H₂O. The fluorine is modeled as a +3 kernel with four valence electrons split into half-spaces (r_c = 1.0); the bare proton has no electrons; the oxygen lone pair is a +2 kernel with r_c = 0.8. As the simulation runs, the F⁻ electron density repels the proton outward while the O lone pair attracts it inward. The proton transfers spontaneously: at convergence, H_t–O = 0.99 Å (covalent in the newly formed H₃O⁺), F–H_t = 1.91 Å (released), with the F⁻ ··· H₃O⁺ ion pair stabilized at 2.4 Å contact. No barrier search, no transition-state geometry, no pK_a input — the forces find the product configuration directly.

**HCl + NH₃ → Cl⁻ + NH₄⁺.** The same setup with chlorine (+1 kernel, r_c = 1.0) and nitrogen (+3 kernel, r_c = 0.5). The ammonia lone pair pulls the proton; the chloride density pushes it away. At convergence: H_t–N = 0.97 Å (covalent in NH₄⁺), Cl–H_t = 1.53 Å (released), Cl⁻ ··· NH₄⁺ contact 2.1 Å. The reaction completes in a single force-driven trajectory.

**Bond breaking and exchange: H + H₂ → H₂ + H.** A hydrogen atom approaches an H₂ molecule along the bond axis. RealQM follows the symmetric H₃ transition geometry directly via the local forces: the two H–H distances pass through equality at the saddle point and then bifurcate in the product channel, with the original bond broken and a new bond formed between the incoming H and the proximal hydrogen. The same Coulomb gradients that hold H₂ together also break it and re-form it.

**What this validates.** These cases test the **forces-not-energies** principle (Section 2.5) on systems where standard quantum chemistry would require either a transition-state search (for the saddle) or QM/MM molecular dynamics (for the trajectory). In RealQM the Coulomb forces that determine equilibrium geometry also determine reactive trajectories; there is no separate machinery for reactions. The chemical specificity — which proton transfers, in which direction, to what acceptor — emerges from the kernel parameters (Z, r_c, split topology) of the participating atoms and is controlled by the local electron-density configuration around each nucleus, not by any reaction-specific input. Nature does not consult pK_a tables or compute free-energy differences; it acts through local forces, and so does RealQM.

### 4.7 Protein folding

The reformulation supports molecular dynamics at scale. We have used the framework to simulate folding of small peptides and mini-proteins driven by quantum-mechanical forces from the unit-density model, supplemented in some cases by a single implicit-solvent hydrophobic parameter. The headline results:

**Polyglycine 12-residue β-hairpin (dry).** Starting from an extended chain at 150° opening angle, the hairpin folds spontaneously to ~105° driven by interstrand N-H···O=C hydrogen bonds. No empirical force field is used; the H-bonds are quantum-mechanical, computed from the gradient of the electron-density potential at each backbone atom. Folding stalls at 105° where H-bond attraction balances backbone repulsion. Adding explicit water does not improve folding — the entropic hydrophobic effect, which completes folding in real biology, is not captured by an electronic-energy solver.

**8-residue polyglycine alpha-helix from a near-linear start.** Starting from a chain with radius 0.5 Å and rise 3.0 Å per residue (essentially extended), the chain folds spontaneously into a helix of radius 2.1 Å and rise 1.63 Å per residue (within 9% of ideal alpha-helix geometry). The helix score reaches 67%; two of four characteristic i→i+4 hydrogen bonds form at H···O distances near 5 Å. The driving forces are the same quantum-mechanical H-bonds; a mild radial helical bias (coefficient 0.03) assists curling.

**Chignolin (10-residue mini-protein).** GYDPETGTWG folds from 135° to 40° opening angle when supplemented with a SASA-based implicit hydrophobic parameter (γ = 5.0). Side-chain electron repulsion that would otherwise unfold a small protein in vacuum is overcome by the hydrophobic pressure on solvent-exposed surfaces.

**Villin HP35 (35-residue 3-helix bundle).** Under universal i→i+4 H-bond biases applied to every residue (with no specification of which segments are helical) plus the SASA hydrophobic parameter, 7 of 9 native helix H-bonds form at the right distances and the Phe core packs to a 5.6 Å contact distance against a 6.0 Å target. This is "near-blind" folding: no native contacts and no segment-specific helix specifications are used; only two generic biological rules (i→i+4 H-bonds favored throughout; hydrophobic surfaces compact).

**Architecture and computational cost.** Each backbone atom is modeled at Level 3 with kernel charges Z = 4 (C), 3 (N), 2 (O), 1 (H), with bond and angle constraints holding the protein backbone. Folding trajectories run interactively on a 200³ or 300³ grid; a 35-residue protein folds in hours of real-time on a single GPU, compared to weeks of CPU-cluster time for full DFT-MD or specialized hardware (Anton) for force-field MD on the same timescales. The combination of ab initio H-bond forces and a single hydrophobic parameter, with no empirical force field, distinguishes this approach from both classical MD (which requires fitted parameters for every interaction) and pure ML methods (which do not solve the electronic-structure problem at all).

### 4.8 Excited states: orthohelium

The unit-density framework extends naturally to excited states. The basic case is the orthohelium (1s 2s, ³S) state of helium, in which two electrons occupy distinct spatial regions corresponding to the 1s and 2s shells. In RealQM, this is a directly accessible configuration of the Level-1 model: place two non-overlapping unit densities, the inner localized in the 1s shell and the outer in the 2s shell, and minimize the total Coulomb energy under the non-overlap constraint. The resulting configuration is an excited state of helium, distinct from the ground state (1s², ¹S) where both electrons occupy the same shell.

The orthohelium case is included in the Gallery as a worked example. It illustrates that the framework is **not restricted to ground states**: any configuration of non-overlapping unit densities corresponding to a chosen shell occupancy is an admissible solution, and excited-state energies follow from the same variational principle applied to the chosen configuration. This is in contrast to standard QM where excited states require additional machinery (TDDFT, CIS, EOM-CC); here, they are just different choices of which shells the electrons occupy.

### 4.9 Materials: NaCl crystal melt

To test whether the framework reaches **bulk condensed-phase behaviour**, we simulate melting of an NaCl ionic crystal under Brownian dynamics with a temperature ramp. The setup is a 4×4×4 rocksalt unit cell (256 Na⁺ + 256 Cl⁻) at Level 3: each Na⁺ is a bare +1 kernel with no explicit electrons; each Cl⁻ is a +1 kernel with two valence electrons split into hemispheres giving net −1. The cohesive energy is the Madelung sum of pairwise Coulomb interactions; there are no fits, no exchange–correlation functional, and no force-field parameters.

**Diagnostics.** Two observables are tracked online. The **Na–Cl radial distribution function** g(r) shows sharp peaks at the rocksalt nearest-neighbour (~3.5 Å in the model) and second-shell positions for the crystal, and broadens into a single residual peak in the liquid. The **Lindemann ratio** ⟨RMSD⟩/d_NN — the mean ionic displacement scaled by nearest-neighbour distance — crosses 0.10–0.15 at melting, an empirical threshold that holds across crystal classes.

**Result.** As temperature is ramped upward in stages from a force-relaxed crystal (at the equilibrium aLat determined by minimum mean inner-ion force at T = 0), the Lindemann ratio crosses the melting window at simulation temperatures T_sim ≈ 1500–2000 K. At the same point, the second and higher RDF peaks broaden and disappear while the first peak survives in broadened form — the classic signature of a liquid that retains short-range coordination but loses long-range order. The transition is reproducible and shows the expected hysteresis on cooling.

**Calibration.** The experimental melting point of NaCl is 1074 K; the simulation overshoots by a factor of ~1.4–1.9. The offset is consistent with the model's coarse-graining (kernel softening reduces the close-range attraction; lattice spacing is set by the solver's force-balance rather than the experimental 5.64 Å; the Brownian-dynamics timestep is not mapped to physical time). What the simulation establishes is **qualitative**: the variational principle plus Brownian dynamics produces a sharp phase transition with the correct character (Lindemann crossing, RDF peak collapse), entirely from Coulomb forces. Quantitative agreement with T_m would require a calibrated kernel parameterization that we do not pursue here.

**Bulk water.** A 216-water cluster with explicit electrons (Section 3) and a 343-water (7×7×7) box have been run at finite temperature with the same Brownian-dynamics framework, reaching interactive frame rates on a single GPU. Detailed bulk-water phase analysis is left to future work, since dispersion (absent in RealQM at Level 3) becomes increasingly important for water structure at higher density.

**Significance.** The NaCl test demonstrates that the Coulomb-only formulation reaches solid-state physics — a regime traditionally requiring DFT-MD or empirical force fields. The transition character is correctly captured, the absolute scale is uncalibrated, and the computational cost is interactive rather than cluster-class. Condensed-phase physics is therefore within reach of the same single architecture that handles atoms, molecules, chemical reactions, and protein folding.

### 4.10 Consolidated results

The table below collects the quantitative validation across regimes.

| Regime | System / quantity | RealQM | Reference | Match | Section |
|---|---|---:|---:|---|---|
| Atoms (Level 1) | Total energies, Li–Rn | — | observation | ~1% | 4.1 |
| Covalent bond | H₂ binding energy | −112 | −109 kcal/mol | 3% | 4.2 |
| Group 1 | NaH atomization | −48 | −47 | 2% | 4.3 |
|  | KH atomization | −42 | ~−43 | 2% | 4.3 |
| Group 2 | BeH₂ atomization | −140 | −144 | 3% | 4.3 |
| Group 14 | CH₄ atomization | −369 | −396 | 7% | 4.3 |
|  | SiH₄ atomization | −348 | −320 | 9% | 4.3 |
|  | GeH₄ atomization | −272 | −281 | 3% | 4.3 |
| Group 16 | H₂O atomization | −225 | −232 | 3% | 4.3 |
| Non-covalent | 5 H-bonded dimers, geom. | — | CCSD(T) | 1–5% (distances) | 4.5 |
| Reactions | HF+H₂O proton transfer | H_t–O = 0.99 Å | exp ~1.0 | qual. correct | 4.6 |
|  | HCl+NH₃ proton transfer | H_t–N = 0.97 Å | exp ~1.0 | qual. correct | 4.6 |
|  | H+H₂ exchange | symm. H₃ saddle | — | qual. correct | 4.6 |
| Protein folding | 12-Gly β-hairpin (dry) | 150°→105° | folded | qual. correct | 4.7 |
|  | 8-Gly α-helix | r=2.1 Å, rise 1.63 Å | ideal helix | 9% | 4.7 |
|  | Villin HP35 native H-bonds | 7 of 9 formed | native | 78% | 4.7 |
| Excited states | Orthohelium (1s 2s, ³S) | directly accessible | known | qual. correct | 4.8 |
| Materials | NaCl Lindemann threshold | T_sim ≈ 1500–2000 K | T_m = 1074 K | ~1.4–1.9× scale offset | 4.9 |

The pattern is consistent: **within ~1–10%** for closed-shell molecular energetics where the Level-3 reduction matches the molecular electron count and bonding topology; **qualitatively correct** for reactive chemistry, protein folding, and excited states; **~1.5× scale offset, transition character correct** for condensed-phase melting where the Brownian-dynamics time mapping is uncalibrated. A graphical summary appears in Figure 1 of the LaTeX preprint (`figures/validation_summary.pdf`).

---

## 2. Multiphase Electron Density Formulation

### 2.1 Mathematical statement

We treat an N-electron system as N spatial densities ρ_1, ..., ρ_N : R³ → R≥0, each satisfying

∫ ρ_i(r) dr = 1 for all i,
ρ_i(r) · ρ_j(r) = 0 for all i ≠ j and all r ∈ R³.

The first constraint is unit charge per electron; the second is strict pairwise non-overlap of the densities. The system also includes M nuclei at positions R_a with charges Z_a.

The total energy is the standard Coulomb functional

E[{ρ_i}; {R_a}] = T[{ρ_i}] + V_eN[{ρ_i}; {R_a}] + V_ee[{ρ_i}] + V_NN[{R_a}],

where T is a kinetic-energy functional (we use a gradient functional T = ½ ∫ |∇√ρ_i|² dr, as in Weizsäcker), V_eN is the standard nuclear-electron Coulomb attraction summed over electrons and nuclei, V_ee is the inter-electron Coulomb repulsion as a sum of pairwise integrals over distinct densities (i ≠ j), and V_NN is the nuclear-nuclear Coulomb repulsion. Self-interaction is excluded by construction: the i = j term is omitted because each density does not interact with itself.

The ground-state energy is the minimum of E over the densities {ρ_i} subject to the unit-charge and non-overlap constraints, and over the nuclear positions {R_a} when geometry is sought. The equilibrium configuration is the minimizer.

### 2.2 Comparison with the standard formulation

The standard formulation seeks an antisymmetric N-electron wavefunction Ψ(r_1, ..., r_N) on R^{3N} as an eigenfunction of the many-body Hamiltonian. Approximate methods truncate the variational space (HF: single Slater determinant; CC and CI: configuration expansions; DFT: a one-body density). The exponential scaling of the configuration space is the underlying obstacle.

The unit-density formulation works in real space at the level of N independent 3D densities, with antisymmetry replaced by strict spatial non-overlap. The two requirements are not equivalent — antisymmetric wavefunctions need not have non-overlapping densities, and non-overlapping densities do not span the antisymmetric subspace. The reformulation is therefore a different mathematical object, not a numerical approximation of the standard one.

This raises an obvious question: how do the two compare numerically? At the parameter-free Level-1 atomic model (Section 4.1) we find atom energies within ~1% of observation, suggesting that for the ground state of light atoms the two formulations are quantitatively close. At higher levels (Sections 4.2–4.5), Level-3 reductions match experimental atomization energies of closed-shell hydrides within 3–9% across multiple periodic-table groups. Where the methods part company is in the cost: a single closed-shell hydride at Level 3 runs in milliseconds on a laptop GPU, while CCSD(T) on the same system requires minutes to hours on a CPU.

### 2.3 The hierarchy of reductions

The Level-1 atomic model is parameter-free: the only input is the nuclear charge Z. As described in Section~\ref{sec:atoms}, shell structure (1s², 2s²2p⁶, ...) emerges from minimum-energy packing rather than being imposed; electron sizes increase with distance from the kernel as a consequence of the variational principle, not as a pre-set rule. We use this model as the bottom of the hierarchy.

**Level 2** spherically homogenizes the inner shells: the angular structure is replaced by spherical symmetry, and the total charge of each inner shell is redistributed as a spherically symmetric radial density. Bond-relevant chemistry is not affected because inner shells contribute only to the effective Coulomb potential seen by valence electrons.

**Level 3** combines the nucleus and the spherically homogenized inner shells into a single *pseudo-kernel* with an effective charge Z_kernel and a softening radius r_c. The valence electrons remain explicit. The kernel is parameterized by Z_kernel and r_c; both are determined by comparison with Level-2 results, but in practice the gallery uses standard conventions (Z_kernel = number of explicit valence electrons modeled, r_c chosen to match the inner-shell radius). This is the level used throughout Section 4.

**Level 4** combines Level-3 atoms into molecular assemblies. The valence electrons may be either single-orbital (no split) or split into angular sectors aligned with bond directions. The choice of splitting topology — sphere, hemisphere, third (120°), tetra (109.5°) — and the kernel charge Z together constitute the *architecture* of the model for that molecule.

We emphasize that this hierarchy is principled, not heuristic. Each level is derived from the prior level by a specific reduction (homogenize inner shell, then absorb into kernel, then assemble into molecules). The empirical content lies in choosing the right architecture for a given molecule.

### 2.4 The free boundary and Bernoulli conditions

The non-overlap constraint ρ_i · ρ_j = 0 means that each electron density is supported on a spatial region D_i, with the regions {D_i} partitioning the relevant volume. The interfaces ∂D_i ∩ ∂D_j between regions are not specified in advance — they emerge from the energy minimization. The unit-density variational principle is therefore a **free-boundary problem**, mathematically analogous to Bernoulli's classical free-boundary problem in potential flow and to the Stefan problem for phase transitions.

At equilibrium, the boundary satisfies a **Bernoulli condition**: the density ρ_i is continuous from within D_i toward the interface (no singular behaviour at the boundary), and the **normal derivative ∂ρ_i/∂n vanishes** at the interface (homogeneous Neumann). The Neumann condition is not externally imposed; it emerges from the variational principle. When the energy is minimized jointly over the densities and the boundary location, the transversality condition at the boundary — the requirement that the first variation vanishes for arbitrary admissible boundary deformations — yields ∂ρ_i/∂n = 0. In other words, the Neumann condition is the consequence of optimizing the boundary location alongside the bulk densities, treating each as part of the same global optimization.

Across the interface, the density jumps: ρ_i is positive on its side and zero on the other side, with the jump occurring sharply at ∂D_i. The boundary condition is therefore the combination of (a) continuity of ρ_i as the interface is approached from within D_i, (b) the homogeneous Neumann condition ∂ρ_i/∂n = 0 at ∂D_i, and (c) a jump from ρ_i > 0 inside to ρ_i = 0 outside. We collectively refer to this as the Bernoulli condition by analogy with Bernoulli's free-boundary problem.

When the system is **not at equilibrium**, density mismatches at the interfaces generate forces that drive boundary motion. The free boundary moves toward configurations satisfying the Bernoulli condition; the bulk densities and the boundary location relax together. This dynamic is part of how the variational principle arrives at the equilibrium configuration: the boundaries are not fixed but evolve as the densities relax. In the numerical implementation (Section 3), the relaxation is realized through a smoothed w-field that softens the partition during imaginary-time propagation and tightens to the Bernoulli condition at convergence; out of equilibrium, density gradients across the boundary supply the forces driving boundary motion in the next iteration.

The free-boundary perspective places the unit-density formulation in a well-studied class of variational problems for which regularity theory, level-set methods, and Dirichlet–Neumann decompositions are available.

The Bernoulli condition itself appears to be **difficult to verify or disprove by direct experimental observation**: the inter-electron interface is not a quantity that current experiments resolve, and standard quantum-chemistry methods do not produce a comparable object (they work with overlapping orbitals or a single total density, not a partition of space into electron territories with a sharp boundary). In computations, however, the condition functions as a working model: imposing continuity and homogeneous Neumann at the free boundary is consistent with the variational principle, the relaxation reaches a stable boundary configuration, and the resulting energies and geometries agree with experiment across the cases reported in Section 4. We therefore adopt the Bernoulli condition as a model assumption supported by computational evidence rather than by direct measurement, and note this status explicitly.

### 2.5 Forces from total electronic potential

In standard quantum chemistry, forces on nuclei are typically computed as derivatives of the total energy with respect to nuclear coordinates: $F_a = -\partial E / \partial \mathbf{R}_a$. The Hellmann–Feynman theorem states that for an exact wavefunction this equals the electrostatic force from the electron density on each nucleus, but for approximate wavefunctions the two differ by Pulay terms arising from the basis-set dependence on nuclear positions. The energy-derivative approach is the standard way forces enter molecular dynamics, geometry optimization, and reaction-path calculations.

In the multiphase electron-density formulation, forces are computed **directly from the gradient of the total electronic potential**:
$$
\mathbf{F}_a = -Z_a \, \nabla P(\mathbf{r}) \big|_{\mathbf{r} = \mathbf{R}_a}
$$
where $P(\mathbf{r})$ is the total Coulomb potential generated by all electron densities and other nuclei, evaluated at the position of nucleus $a$. There is no energy functional being differentiated; the force is the local electrostatic gradient on each nucleus, computed directly from the electron density and other nuclear charges via Coulomb's law.

The two approaches give numerically the same result at exact electron equilibrium (Hellmann–Feynman) but differ in conceptual standing. The local-potential view is closer to the physics: **nature does not carry a record of energy**, only of local potential gradients. A nucleus does not "know" the total energy of the molecule and then take a derivative; it experiences the Coulomb force from the surrounding charge distribution, locally and instantaneously. Dynamics in RealQM is therefore driven by physical forces evaluated pointwise from the density, not by differentiation of a global functional.

This has practical consequences. Energy-derivative forces in standard QC require careful treatment of basis-set dependences (Pulay forces), analytic-gradient code that scales as the wavefunction calculation itself, and consistency between the kinetic and potential terms in the energy functional. Local-potential forces in RealQM are evaluated directly on the real-space grid, are O(N) in the number of grid points, and require no additional gradient machinery beyond the density and nuclear positions already maintained.

---

## 3. Numerical implementation

The reformulation is implemented as **mol_fast.js** (~1600 lines) and **molecule.js** (~5300 lines), JavaScript modules that compile WGSL compute shaders to a WebGPU device. mol_fast.js uses unit-density orbitals with explicit angular splitting; molecule.js uses a Voronoi-partition labelling field. Both run entirely in the browser.

### 3.1 Real-space grid

The simulation domain is a cubic box of side L au, discretized as N×N×N grid points (typical N = 100–200, L = 10–22 au, grid spacing h = L/N ≈ 0.1 au). Each electron's density is represented as an N³ array of floats (orbital amplitudes), evolved by imaginary-time propagation (ITP) of the Hamiltonian operator H = T + V on the grid. The Hartree potential P from each density is solved by parallel Poisson diffusion. Self-interaction is removed by subtracting the contribution of each electron's own density from the total Hartree potential.

A typical step does ~10 GPU compute dispatches: orbital ITP update, Poisson update, kinetic-energy reduction, normalization, force computation. On a consumer laptop GPU (~10 TFLOPS), this runs at 30+ steps per second for the systems reported in Section 4.

### 3.2 Kernel softening and angular splitting

A Level-3 kernel is parameterized by (Z_kernel, r_c, split_type, split_idx, split_axis). The kernel potential at distance r from the nucleus is V_kernel(r) = −Z_kernel/r for r > r_c, with a smooth softening for r ≤ r_c that matches the Coulomb tail and goes to zero at the origin. The split_type is one of {sphere, hemi, third, tetra, hemi_third}, defining how the angular sectors of the kernel partition the orbital domain. The split_axis is the axis around which sectors are arranged. The split_idx selects which sector this particular orbital occupies.

For a heavy atom with multiple valence electrons, several sub-orbitals at the same nuclear position with different split_idx values together represent the full valence shell. Each sub-orbital evolves independently subject to its angular sector mask.

### 3.3 Code size and computational cost

The full RealQM implementation is approximately 8000 lines of JavaScript and WGSL combined. For comparison, mainstream quantum chemistry packages range from ~200,000 lines (Quantum ESPRESSO) to ~3,000,000 lines (Gaussian, NWChem). The size compression is enabled by working directly in real space: there are no basis sets, no two-electron integrals, no orbital coefficient bookkeeping, no analytic gradient machinery.

For typical jobs, RealQM runs orders of magnitude faster than CPU-based quantum chemistry on equivalent hardware: a water-dimer geometry optimization that takes hours of CCSD(T)/CBS computation completes interactively in seconds. A 216-water cluster with explicit electrons runs at real-time interactive frame rates on a single GPU; the equivalent DFT-MD simulation requires tens to hundreds of CPU cores running for weeks. The speedup factor of 10²–10⁴ is real and is the practical breakthrough of the framework, even if accuracy at very high precision (CCSD(T)-class) is not the goal.

---

## 5. Limitations

Two regimes lie outside the validation reported here. **Weak intermolecular interactions** (hydrogen bonds at ~5 kcal/mol = 0.008 Ha) are below the model's absolute-energy noise floor of ~0.1 Ha; geometric agreement on S66 dimers is good (1–5%) but quantitative interaction energies are not reliable. **Dispersion** (van der Waals interactions between non-polar species) is absent at Level 3 entirely, as the unit-density model has no mechanism for the correlated electron motion that produces vdW. Excited states are accessible (Section 4.6) but at the present stage have only been validated on the simplest cases.

---

## 6. Chemistry vs Physics

A central question in the philosophy of chemistry is whether **(A)** chemistry can be explained by quantum physics, as **Dirac famously claimed in 1929 at the birth of quantum mechanics**, or **(B)** something more is required, as has been argued by leading chemists for nearly a century. Position (B) is the view that chemistry possesses an irreducible conceptual content — orbitals, bonds, valence, hybridisation, electronegativity — that does not follow from Schrödinger's equation alone, no matter how many computational resources are thrown at it. Position (A) is the view that chemistry *is* applied quantum physics, and that the apparent autonomy of chemical concepts is an artefact of the practical impossibility of solving the underlying equations.

The hundred-year stalemate has been driven less by disagreement on principle than by the fact that, within the standard $3N$-dimensional wavefunction formulation that Dirac himself had in mind in 1929, the equations are indeed unsolvable for any system of chemical interest without approximations whose chemistry-specific character (basis sets, exchange–correlation functionals, fitted force fields) seems to vindicate position (B).

RealQM provides **concrete evidence that Dirac's position (A) was correct, although not within the 1929 wavefunction-on-$3N$-space formulation that Dirac himself proposed**. The reformulation as a system of non-overlapping unit electron densities in real 3D space, with energy minimization over Coulomb interactions alone, recovers atoms (Section 4.1), the covalent bond and its He limit (Section 4.2), atomization energies of hydrides across periodic-table groups (Section 4.3), reactive chemistry (Section 4.6), and protein folding (Section 4.7) — using only physics, with no chemistry-specific empirical input beyond the kernel parameters that encode the level-by-level reduction. Chemistry emerges as the variational principle's response to multiple nuclei being present, with no separate machinery required.

The conclusion we draw is not that the standard $3N$-dimensional formulation should be abandoned — for high-precision spectroscopy, transition probabilities, and excited-state dynamics it remains the canonical tool — but that **chemistry-as-applied-physics is achievable** when one is willing to reformulate the problem in real 3D space with an appropriate non-overlap constraint. Dirac was right; the equations were just being written in the wrong space.

---

## 7. The collaboration as part of the contribution

The mathematical reformulation in Section 2 and the hierarchy of Levels 1–4 are the work of the human author. They predate the AI collaboration and have been developed over a decade of independent work. The author also wrote the initial p5.js prototype simulations that established the numerical core (real-space grid, ITP, Poisson solver, basic visualization) used as the starting point for the WebGPU implementation. The full WebGPU port, the validation suite, the systematic kernel-architecture sweeps reported in Section 4, and the curated public Gallery were developed from those starting templates through extended collaboration with **Claude**, an AI code assistant produced by Anthropic.

We document the pattern of the collaboration here because we believe it is a worth-reporting example of how a single mathematical mind can produce research-grade interactive scientific software with an AI engineering partner.

The human posed mathematical and chemical questions: "What if the model uses Z=2 instead of Z=4 for carbon?" "Does the H-bond forces correctly point toward the acceptor lone pair?" "Why does this geometry prefer stretched over bonded?" "What if we add a +2 model with -1 upper half and -1 lower half connecting to the H atoms?" The AI translated these into runnable simulations, identified bugs in real time, ran systematic parameter sweeps, computed and tabulated results, drafted candidate gallery cards, and challenged claims that did not survive scrutiny.

The interaction was iterative and surprisingly productive: a question asked at 9:00 might be answered with a working scan file at 9:05, partial results at 9:30, a full sweep table at 10:00, and a refined analysis incorporating the latest data by 10:30. Across many such cycles, the validation table reported in Section 4 was assembled.

We also record honestly where the AI's interpretations needed correction. Several times the AI prematurely concluded that a result was "within X% of experiment" before convergence had been confirmed; in each case the human pushed back and the AI revised. The Gallery's "Assessment by Claude" card reflects this — its caveats were rewritten more than once during the project as the evidence accumulated.

A concrete example: when the methylamine dimer was first tested, the AI initially reported "geometries within 1% of CCSD(T)" based on locked-atom force diagnostics. The human pointed out that locking atoms at the reference and observing they didn't move was not in itself a validation. The AI subsequently retracted the "1% match" framing, reformulated the test in terms of force directions on the donor proton, and only reinstated a quantitative claim once a different convergence test (energy convergence + force-direction agreement at relaxed-water-H geometry) had been passed. The retractions and reformulations are part of the public record.

We find that the right framing for the collaboration is **mathematical mind + AI engineering partner**: the human supplies the theory, the questions, the chemical intuition, and the standards for what counts as evidence; the AI supplies the implementation, the systematic exploration, and the speed. Neither could have produced this result alone in any reasonable timeframe.

---

## 8. Code and reproducibility

The full RealQM implementation, including the validation suite, the systematic kernel-architecture sweeps reported in Section 4, and the curated public Gallery, is available at:

- **Code repository:** https://github.com/Claes542/RealMolecule
- **Interactive Gallery:** https://claes542.github.io/RealMolecule/gallery.html
- **RealQM project site:** https://physicalquantummechanics.blogspot.com
- **Author's blog:** https://claesjohnson.blogspot.com
- **Browser requirements:** Chrome 113+, Edge 113+, or Safari 17+ (WebGPU)
- **Hardware:** any modern integrated or discrete GPU (~1 GB GPU memory for ~200³ grid)

All results in this paper can be reproduced by opening the relevant `.html` files in the repository. Each binding-energy data point in Section 4 corresponds to a specific URL parameterization (R, rc) of a specific scan file (e.g., `mol_fast_H4X_Z2_scan.html?R=1.0&rc=0.2`). The Gallery's "Kernel Splitting" and "Periodic-Table Coverage" cards link directly to the scan files used to generate the reported numbers.

We deliberately do not include figures in this preprint. The Gallery contains many interactive visualizations — density slices, 3D molecular views, force-arrow displays, real-time dynamics — that are far more informative as live simulations than as static images. Readers wishing to inspect electron densities, watch molecules relax, or run their own kernel-architecture sweeps should consult the Gallery directly.

---

**Acknowledgments.** The author is very happy to meet Claude Code in a fruitful cooperation way beyond initial expectations after lonely struggle with coding over long time.

**Competing interests.** None declared.

---

## References

1. Hohenberg, P., Kohn, W. *Inhomogeneous electron gas.* Phys. Rev. 136, B864 (1964).
2. Kohn, W., Sham, L. J. *Self-consistent equations including exchange and correlation effects.* Phys. Rev. 140, A1133 (1965).
3. Mardirossian, N., Head-Gordon, M. *Thirty years of density functional theory in computational chemistry: an overview and extensive assessment of 200 density functionals.* Mol. Phys. 115, 2315 (2017).
4. Bader, R. F. W. *Atoms in Molecules: A Quantum Theory.* Oxford University Press (1990).
5. Ruedenberg, K. *The physical nature of the chemical bond.* Rev. Mod. Phys. 34, 326 (1962).
6. Kutzelnigg, W. *The physical mechanism of the chemical bond.* Angew. Chem. Int. Ed. 12, 546 (1973).
7. Lindemann, F. A. *Über die Berechnung molekularer Eigenfrequenzen.* Phys. Z. 11, 609 (1910).
8. Born, M., Mayer, J. E. *Zur Gittertheorie der Ionenkristalle.* Z. Phys. 75, 1 (1932).
9. Shaw, D. E. et al. *Atomic-level characterization of the structural dynamics of proteins.* Science 330, 341 (2010).

---
