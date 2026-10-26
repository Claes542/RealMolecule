# RealQM: the open formulation choices

**Why this document.** The computing has stopped being the bottleneck. What now limits the
programme is that the construction is **underdetermined**: at several points a calculation requires
a choice that the theory does not fix, so two people following "RealQM" can compute different
numbers without either making an error. Each such point is a place where no amount of machine work
settles anything, because the machine can compute every branch and cannot tell you which branch is
the theory.

What follows is an inventory. For each choice: what the alternatives are, what the stated theory
says, what has been measured, which results depend on it, and what would close it. Numbers are from
the repo and are cited to the file that produced them.

---

## 1. The overlap dial

**The choice.** Strict non-overlap (disjoint domains) against partial overlap.

**What the theory says.** `RealQM_method.tex:101` — "{Omega_i} is a partition of physical space into
open **non-overlapping** regions, Omega_i ^ Omega_j = empty for i != j"; line 138 — "Pauli exclusion
through non-overlap... distinct electrons occupy **disjoint** domains". `FoundChem.tex:331` — "In
RealQM electrons do not overlap."

**What is done in practice.** Overlap, in three places. `FoundChem.tex:209` initialises "allowing
the free boundary to develop dynamically with an **overlap of level set functions**". The
homogenised shell — two electrons sharing one density with self-repulsion reduced by
`(N_s-1)/N_s` — is full overlap, and is what the atom calculations use. `mol_fast.js` overlaps
domains as a fast approximation.

**What is measured.** Both ends bracket the truth. Helium: homogenised **-2.845**, disjoint
hemispheres **-3.026**, exact **-2.9037** — the exact value lies *between*, and moving from one end
to the other *crosses* it rather than converging (`tools/axisym_realqm.py`, recorded in
`project_axisym_solver`).

**What depends on it.** Every multi-electron shell; helium's energy; the legitimacy of the empty
core (which sits inside the dial, not outside it).

**What would close it.** A statement in the theory of where on the dial a given configuration sits,
derived rather than calibrated. At present the position is set implicitly by whichever construction
a given code happens to use.

---

## 2. H^1 continuity against masked interfaces

**The choice.** Whether the kinetic form is assembled across a domain interface or not.

**What the theory says.** Unambiguous: Psi in H^1(R^3), "the psi_i being continuous across joint
boundaries Gamma_i ^ Gamma_j" (`RealQM_method.tex:101`). So the straddling faces are assembled.

**What is done in practice.** Both. The radial atoms solver is already correct — `atom_headless.js`
has ONE global array `u` partitioned by shell boundaries `M[s]`, with the Laplacian running across
them. But `molecule.js` masks, in 18 lines across 3 shaders (`700-705, 800-805, 2135-2140`):
`select(uc, Ui[nbr], l_nbr == myL && !excl)` substitutes the cell's own value whenever the
neighbour carries a different label. `atom.js` masks in its **energy** (line 262, "Energy: only
count where this electron's W dominates") while its **operator** couples across the interface via
`div(W grad U)` with face weights — so one file does it two ways.

**What is measured** (`tools/axisym_h1.py`, controlled swap, one line apart):
- at a **symmetry plane** the two agree to **4e-8** — so symmetric systems cannot test this, and
  He and H2 split on the midplane are null tests;
- at an **asymmetric** interface, masking under-stiffens by ~5x AND has **no continuum limit**
  (0.1296 -> 0.1472, still climbing) where H^1 converges to **0.739**;
- with a **hard** mask on a sharp partition the interface is destroyed, not shifted: R* runs to the
  scan bound with E = -54.7 against a stable interior R* = 0.446.

**What depends on it.** Anything with an asymmetric interface: molecules, core/valence, shear,
defects. Not the alkali cell face, which is symmetric.

**What would close it.** Nothing in the theory is unclear here — this one is an implementation debt,
not a formulation gap. The port of the unmasked stencil plus per-domain charge constraints to the 3D
solvers is the outstanding work.

---

## 3. The core model

**The choice.** Hard depletion (density zero inside R) against the empty core (V=0 inside r_c,
electron admitted) against a screened kernel.

**What the theory says.** It prescribes territorial exclusion at R. It does not prescribe the empty
core, which admits the density and only cancels the attraction.

**What is measured.** Hard depletion is **unusable in a cell**: Neumann at both the core boundary
and the cell face makes a constant density an exact kinetic null mode, **4.3e-10** against the
atom's **0.617**, structural across n = 500..8000 (`project_double_neumann_null_mode`). So the cell
is forced to the empty core, with `r_c = 0.855 R` as the conversion between the two radii. And the
empty core is not an approximation to the screened potential: at r = 0.1 the true screened potential
is **-28.25 Ha** where the model says 0, with a discontinuity of 1/r_c = **0.583 Ha** at r_c. It is
a surrogate for the exclusion, not a model of the potential.

Also measured: a **smooth** one-parameter core, `-(1 - exp(-(r/lam)^3))/r`, fitted to the same
ionization energy, changes nothing material -- r_s +17.8% against +20.0%, B -33.5% against -30.5%.
Only 6.6% of the valence charge sits inside r_c, so the interior shape is weighted lightly. **The
core model is not where the residual error is.**

**What depends on it.** Every solid-state result, and the determination of r_c.

**What would close it.** Either a derivation of where on the dial (choice 1) the core sits, or a
demonstration that the cell admits exclusion after all under a construction that avoids the null
mode.

---

## 4. Domain topology for a shell

**The choice.** For a shell holding k electrons: k one-electron domains, or one multi-occupancy
domain with the `(N_s-1)/N_s` reduction. And if k one-electron domains, arranged how -- nested
radially, or packed angularly.

**What the theory says.** `CLAUDE.md` states the house construction: splitting a -Z valence into Z
one-electron domains is exact, with n(n-1)/2 pairs counted once by the i!=j rule, no (Z-1)/Z factor.
"A -Z multi-occupancy blob is the exception and needs justifying." The atom calculations use the
exception.

**What is measured.** Topology dominates. For lithium's core pair, **radially nested** shells
misplace the electrons -- "they sit side by side at the same radius, not one outside the other" --
and cost **28% of the total energy**, where the homogenised shell reproduces it to **0.03%**
(`realqm_elasticity.tex:259`). Independently: a sharp radial partition built this way gives
R = 0.446 and 64% over-binding against the working code's 2.000 and good totals
(`tools/li_cell_hom.py`, shelved for exactly this reason).

**What depends on it.** Every atom beyond helium; the interface radius R = 2.00 on which lithium's
parameter-free chain rests.

**What would close it.** A rule fixing the packing from Z alone. The pair-and-octet / 4+4 result is
the candidate; whether the interface survives an angular partition of the core is stated as untested
in `realqm_elasticity.tex:258` and remains so -- the attempt needs a cusp-correct formulation in a
geometry admitting angular domains, which neither existing tool provides.

---

## 5. Kernel softening and the cusp

**The choice.** How the nuclear 1/r is represented near the origin: softening `1/sqrt(r^2+eps^2)`
with eps tied to the mesh, a cell-averaged kernel, a pseudokernel, or the bare cusp in a u = r*psi
formulation.

**What is measured.** Tying eps to h is **mesh-divergent for Z > 1**: with eps = 0.285h the
Li axisym calculation has no minimum and the energy falls monotonically under refinement, and at
fixed R the energy moves -7.804 -> -7.952 between meshes. The radial code instead integrates the
kernel over the cell, giving 0.923*Z/h at the first node rather than Z/h, and documents the cusp as
"THE DOMINANT SOURCE OF IMPRECISION" (`tools/atom_headless.js`). The u = r*psi form with
(1/2)INT(u'-u/r)^2 penalises concentration at the origin correctly and is the stated cure for the
density formulation's collapse.

**What depends on it.** All absolute energies; any calculation at Z > 1 in a psi formulation.

**What would close it.** A single stated prescription for the kernel, independent of the mesh, used
by every code.

---

## 6. The cell electrostatics

**The choice.** `-0.9/r_s` (self-term included) against `-1.5/r_s` (the i!=j rule).

**What is settled, and it is not what it looks like.** The Madelung constant (bcc alpha = 0.895930,
validated to 6e-7 in `tools/elastic3d_shear.py`) is the **TOTAL** lattice energy, including each
cell's own cloud interacting with itself. The i!=j rule removes that self-energy (0.6/r_s for a
uniform unit sphere), giving **-1.49593/r_s** -- i.e. the -1.5 row, to 0.27%. So **-0.9 and -0.896
are both the total**, and their agreement is not evidence for the first row. The construction needs
the i!=j value, which is -1.5.

**Consequence.** B is 216% off under the principled choice and 23% off under the other. The paper
reports the second and calls the modulus unsettled, which is correct. **Do not "fix" this with the
Madelung sum** -- that confuses the total with the i!=j value.

**What would close it.** A derivation of the cell's electrostatic energy from the construction
itself rather than from a decomposition of the neutral-sphere result.

---

## 7. Where r_c comes from

**The choice.** The measured ionization energy; the all-electron atom's own free interface through
the conversion; or a fit to the atomic spectrum.

**What is measured.** For lithium the first two agree to **0.35%** -- 1.716 from the IE directly,
1.710 from the interface at 2.00 through 0.855 -- and these are the two the paper uses. The spectral
fit gives **1.95**, 14% away. Separately, both prescriptions **overshoot** what the solid's own
observables want, systematically and in every element: the IE route by +4.2% to +14.2% (mean +9.5%),
the universal constant by -2.8% to +7.1% (mean +3.1%), against the midpoint of the window where r_s
and B are each reproduced. That window is itself narrow and narrows down the group -- the two
observables want r_c within 8% for Li and **0.3% for Cs** -- so the cell model is internally
consistent and the **transfer** is what overshoots. (All conditional on choice 6.)

**What depends on it.** Lithium's parameter-free claim, and the whole alkali series.

**What would close it.** Density matching rather than radius matching: require the empty-core
valence density to agree with the all-electron valence density outside r_c, with both carrying unit
charge. That fixes r_c with no fitted ratio, replacing 0.855 with a computed number. It needs a
trustworthy all-electron valence density, which is choice 4.

---

## What this adds up to

Choice **2** is implementation debt, not a gap -- the theory is clear and one solver already obeys
it. Choices **1, 3, 4** are a single question seen from three sides: *how much do domains overlap,
and what fixes it?* Choice **5** is a convention that merely needs stating once. Choices **6** and
**7** are downstream of 1-4 and would likely settle with them.

So the formulation gap is **one question, not seven**: the dial has no stated position. Every number
the programme produces inherits a choice made by whichever code produced it, and the machine cannot
close that -- it can compute every branch, and has.
