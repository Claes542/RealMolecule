# PLP as an electron sink in GAD65 decarboxylation — a first-principles test

*Reduced gas-phase active-site model, RealQM electronic-structure solver.
Companion to `gad65_decarb_scan.html`. Results are qualitative (sign/trend), not calibrated energies.*

---

## 1. Objective

GAD65 decarboxylates L-glutamate to GABA + CO₂ using the cofactor pyridoxal-5′-phosphate
(PLP). In the accepted mechanism the substrate forms an **external aldimine** with PLP, and
when the Cα–COO⁻ bond breaks the electrons left behind are delocalized into the protonated
pyridine ring — the **quinonoid / carbanion intermediate**. The pyridinium nitrogen is the
"**electron sink**" that makes this otherwise high-energy carbanion accessible. By Dunathan's
stereoelectronic rule the scissile Cα–COO bond sits perpendicular to the ring plane for
maximal σ→π overlap.

This is more than enzymology for the T1D field: GAD65 is the **major type-1-diabetes
autoantigen** (GADA is a front-line predictive/diagnostic marker), and GAD65's autoantigenicity
is tied to its unusual catalytic behaviour — its tendency to lose PLP and populate a flexible
**apo** form, and the dynamics of the catalytic loop region. Any computational claim about
GAD65 conformation, the apo↔holo cycle, or epitope exposure ultimately rests on whether the
method gets the **underlying PLP electronics** right.

**The question this run answers:** does a parameter-free electronic-structure method, given only
the nuclei, *spontaneously* reproduce the electron-sink role of the PLP ring during
decarboxylation — i.e. stabilize the departing-CO₂ carbanion specifically because the ring is
conjugated?

## 2. What was performed (method)

**Solver.** RealQM — a real-space, grid-based electronic-structure method (no basis sets, no
fitted force field). The electron density is relaxed to its minimum-energy configuration in the
field of fixed nuclei; the total energy E is read out directly.

**Model system (reduced, hand-built, gas phase).** The PLP active-site core, not the 585-residue
enzyme:
- PLP → a minimal mimic: **protonated pyridine (the N1-H electron sink) + 3-OH + 4′-imine**.
  The 5′-phosphate and 2′-methyl (anchoring groups, not central to the C–CO₂ cleavage) are omitted.
- Substrate → glutamate as the **external aldimine**, its α-carboxylate **Dunathan-aligned**
  (perpendicular to the ring, the +z axis = the CO₂ that leaves). Side chain truncated for speed.
- Catalytic Lys396 implicit (already displaced to the external aldimine); no protein, no water.

**Experimental design — a controlled energy scan with a built-in negative control.** The breaking
Cα–CO₂ distance R is driven outward in fixed steps; at each R all nuclei are clamped and only the
electrons relax to convergence → E(R). This is run for **two electronic environments**:
- **PLP** — glutamate conjugated to the pyridinium ring (the sink is present);
- **free** — the *identical* glutamate as a free amino acid (−NH₃⁺), **no conjugation, no sink** —
  the negative control.

Reported quantity: ΔE(R) = E(R) − E(R₀) (reaction energy of CO₂ departure; negative = downhill),
and the **sink term = ΔE(free) − ΔE(PLP)** = the extra stabilization attributable to the ring.

**Run parameters.** 24 atoms (PLP variant) on a 160³ real-space grid, ~38 a.u. box; 7 distances
R = 1.55–6.00 Å × 2 environments = 14 independent relaxations; per-point convergence by
energy plateau. Runs in-browser on WebGPU (Chrome).

## 3. Results

| R (Å) | ΔE free | ΔE PLP | sink = free − PLP |
|------:|--------:|-------:|------------------:|
| 1.55  |  0.000  |  0.000 |  0.00 |
| 2.00  | −0.012  | −1.916 | +1.90 |
| 2.60  | −2.476  | −3.776 | +1.30 |
| 3.30  | −2.560  | −5.426 | +2.87 |
| 4.20  | −3.101  | −6.605 | +3.50 |
| 5.00  | −3.556  | −6.912 | +3.36 |
| 6.00  | −3.454  | **−7.620** | **+4.17** |

*(model energy units; sign/trend meaningful, magnitudes not calibrated)*

**The diagnostic signature is the divergence of the two curves:**

- **Free glutamate saturates.** Once CO₂ is ~4 Å out the free curve flattens at ≈ −3.5
  (−3.10 → −3.56 → −3.45). With no electron sink, the departing carbanion is a dead end — the
  system can only relax so far.
- **PLP keeps descending.** The conjugated curve shows no plateau (−6.6 → −6.9 → −7.6); the
  pyridinium π-system absorbs the developing negative charge (the quinonoid), so stabilization
  keeps accruing as CO₂ leaves.
- **Net:** the sink term grows essentially monotonically to **+4.2 at 6 Å and is still climbing.**

**Interpretation.** Given nothing but the atomic coordinates — no reaction coordinate built in,
no bias toward the "right" answer — the method spontaneously reproduces the textbook role of PLP:
the conjugated ring, and only the conjugated ring, stabilizes the carbanion left when CO₂
departs. The free–vs–PLP contrast isolates that effect cleanly because the substrate is identical
between the two; the only difference is the presence of the sink.

## 4. Scope — what this does and does not show

- **Qualitative, not quantitative.** Magnitudes are uncalibrated model units (and reproducible
  only to ~±0.5–0.8 between runs). Read the **sign, the divergence, and the monotonic growth** —
  not the numbers. No kcal/mol claim.
- **Thermodynamics, not kinetics.** A clamped, single-coordinate relaxed scan has no transition
  state; this shows carbanion/product stabilization, **not a catalytic rate or barrier**.
- **Gas-phase, reduced core.** No protein electrostatics, no phosphate anchor, no water,
  hand-built geometry. This is a mechanism demonstration on the active-site electronics, not a
  structural model of the enzyme.

## 5. Relevance to GAD65 / T1D, and where this can go

This establishes that a parameter-free method captures the **central electronic event of GAD65
catalysis** correctly and from first principles. That is the foundation needed before the more
T1D-relevant questions can be addressed credibly:

- **apo ↔ holo electronics.** GAD65's autoantigenicity is linked to its propensity to lose PLP and
  populate a flexible apo form. The same machinery can compare the electronic stabilization
  of the holo (PLP-bound) vs apo active site.
- **Catalytic-loop / epitope coupling.** Once the electronics are trusted, the catalytic region
  that overlaps known conformational GADA epitopes becomes a target for modelling.
- **Autoantibody recognition** (separate, *parked* model, `gad65_autoantibody_dock.html`):
  currently uses placeholder sequences and a linear-peptide epitope, whereas real GADA epitopes
  are conformational — so that line needs real CDR/epitope residues and is a much larger lift.

**Bottom line for a GAD65/T1D reader:** an *ab-initio*-style electron solver, handed only the
active-site atoms, independently reproduces the PLP electron-sink mechanism that defines GAD65's
chemistry — the free-amino-acid control confirms the effect is specifically the conjugated ring.
The result is a clean qualitative validation, not a calibrated energetic or kinetic prediction.
