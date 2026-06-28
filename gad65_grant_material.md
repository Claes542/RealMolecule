# Grant proposal material — RealQM electronic structure of GAD65: from the PLP electron sink to the type-1-diabetes autoantigen

*Draft building blocks (Summary, Aims, Significance, Innovation, Preliminary Data, Approach, Rigor).
Funder-agnostic; retarget tone/length for NIH R01/R21, JDRF/Breakthrough T1D, Diabetes UK, or an
EU/national methods call. Claims are deliberately calibrated to what the preliminary run actually shows.*

---

## Project summary / abstract

Glutamic acid decarboxylase 65 (GAD65) is the principal autoantigen of type 1 diabetes (T1D):
autoantibodies to GAD65 (GADA) are among the earliest and most predictive markers of disease, and
GAD65 itself remains a leading antigen-specific immunotherapy candidate. GAD65's immunogenicity is
inseparable from its biochemistry — it is an unusually unstable enzyme that readily loses its
pyridoxal-5′-phosphate (PLP) cofactor to form a conformationally mobile **apo** species, and the
flexible catalytic-loop region implicated in this apo↔holo cycle overlaps known conformational GADA
epitopes. Yet the electronic events that drive this chemistry have never been modelled from first
principles at the active site, because conventional quantum chemistry is too costly for reactive
scans in a system this size and force fields cannot represent bond making/breaking or cofactor
electronics.

We propose to apply **RealQM**, a parameter-free real-space electronic-structure method (no basis
sets, no fitted force field, GPU-accelerated) that resolves reactive chemistry cheaply enough to scan
an enzyme active site. In preliminary work RealQM, given only the active-site nuclei, **spontaneously
reproduced the textbook PLP electron-sink mechanism** of GAD65 decarboxylation: with the conjugated
pyridinium ring present the carbanion left by departing CO₂ is progressively stabilized, while an
identical free-amino-acid control (no ring) saturates — the two energy curves diverge cleanly. We
will (Aim 1) put this mechanism on a quantitative footing and extend it to the holo-vs-apo active
site; (Aim 2) couple active-site electronics to catalytic-loop conformation and epitope exposure;
and (Aim 3) build atomistic conformational-epitope models for GADA recognition. The result will be
the first first-principles electronic picture linking GAD65 catalysis, cofactor loss, and
autoantigenicity — a mechanistic foundation for epitope-targeted T1D biomarkers and tolerogenic
immunotherapy design.

## Specific aims

**Aim 1 — Quantify the PLP electron-sink electronics, holo vs apo.**
Convert the qualitative preliminary result into calibrated, reproducible electronic-structure
profiles of the decarboxylation coordinate. Compare the PLP-bound (holo) active site against the
PLP-free (apo) state to test the hypothesis that loss of the electron sink not only abolishes
catalysis but reshapes the local electronic landscape that the catalytic loop responds to.
*Deliverable:* validated holo/apo electronic profiles + benchmark against high-level reference
calculations on a tractable model.

**Aim 2 — Couple active-site electronics to catalytic-loop conformation and epitope exposure.**
Map how the holo↔apo electronic difference propagates to the mobility of the catalytic-loop region
that overlaps conformational GADA epitopes. Test whether the apo electronic state correlates with
increased exposure of the residues recognized by patient autoantibodies.
*Deliverable:* a residue-resolved link between cofactor state and epitope accessibility.

**Aim 3 — Atomistic conformational-epitope models for GADA recognition.**
Replace the current placeholder linear-peptide dock with real anti-GAD65 CDR sequences and the
genuine *conformational* (discontinuous) GAD65 epitope surface, and characterize paratope–epitope
recognition.
*Deliverable:* atomistic recognition models for the major conformational GADA epitopes, ranked by
interface energetics.

## Significance

- **Clinical.** GADA is a cornerstone of T1D risk prediction; understanding *which* conformational
  states present *which* epitopes could sharpen biomarker interpretation and inform GAD65-based
  antigen-specific immunotherapy (a strategy that has shown signal but inconsistent efficacy).
- **Mechanistic gap.** The apo↔holo instability that makes GAD65 (but not the homolog GAD67)
  autoantigenic is documented phenomenologically but not understood electronically. This proposal
  addresses the molecular origin, not just the correlation.
- **Methodological.** A reactive, parameter-free quantum method that can scan an enzyme active site
  on commodity GPUs would be broadly enabling for cofactor enzymology beyond GAD65.

## Innovation

1. **A reactive electronic-structure method at enzyme scale.** RealQM uses no basis set and no fitted
   parameters; it resolves bond making/breaking and cofactor electronics directly, at a cost that
   permits *scans* (here 14 independent relaxations) rather than single points.
2. **Built-in controls by construction.** The free-amino-acid negative control isolates the ring's
   contribution because the substrate is identical between arms — an internal control that
   parameterized methods cannot cleanly provide.
3. **A first-principles bridge from catalysis to autoimmunity.** No prior work connects GAD65's
   cofactor electronics to its epitope landscape at the electronic-structure level.

## Preliminary data

A reduced gas-phase model of the GAD65/PLP active site (pyridinium electron-sink mimic + 3-OH +
4′-imine + Dunathan-aligned glutamate external aldimine; 24 atoms, 160³ real-space grid) was scanned
along the breaking Cα–CO₂ coordinate for two electronic environments — PLP-conjugated vs. an
identical free amino acid (no sink). RealQM relaxed the electron density at each clamped geometry.

| R (Å) | ΔE free | ΔE PLP | sink = free − PLP |
|------:|--------:|-------:|------------------:|
| 1.55  |  0.000  |  0.000 |  0.00 |
| 2.00  | −0.012  | −1.916 | +1.90 |
| 3.30  | −2.560  | −5.426 | +2.87 |
| 4.20  | −3.101  | −6.605 | +3.50 |
| 6.00  | −3.454  | −7.620 | **+4.17** |

*(model energy units; sign/trend interpreted, magnitudes not yet calibrated — see Rigor)*

**The free curve saturates (≈ −3.5 beyond ~4 Å) while the PLP curve keeps descending (to −7.6 at
6 Å); the sink stabilization grows monotonically to ~+4.** Given only coordinates and no built-in
reaction coordinate, the method reproduces PLP's electron-sink role, and the identical-substrate
control attributes the effect specifically to the conjugated ring. *This establishes feasibility for
Aim 1 and motivates the holo/apo contrast.*
(Interactive: `gad65_decarb_scan.html`; full write-up: `gad65_decarb_results.html`.)

## Approach (highlights & milestones)

- **Yr 1 (Aim 1):** tighten convergence for reproducible energetics; benchmark the reduced model
  against high-level QM (e.g. coupled-cluster/DFT on a tractable cluster) to calibrate RealQM model
  units; build the apo active-site model and compute the holo/apo electronic difference.
  *Milestone:* calibrated, reproducible holo & apo decarboxylation profiles.
- **Yr 2 (Aim 2):** embed the active-site electronics in successively larger environments (phosphate
  anchor, key first-shell residues, implicit/explicit water); relate the cofactor-state electronic
  difference to catalytic-loop mobility and epitope-residue exposure.
  *Milestone:* cofactor-state → epitope-accessibility map.
- **Yr 3 (Aim 3):** assemble conformational-epitope models with real GADA CDRs; characterize
  recognition energetics; cross-check against known GADA epitope-mapping data.
  *Milestone:* ranked atomistic recognition models for the major conformational epitopes.

## Rigor, reproducibility & risk mitigation

- **Honest scope of preliminary data.** Current results are *qualitative*: magnitudes are
  uncalibrated model units (reproducible to ~±0.5–0.8 between runs) and the scan captures
  *thermodynamic* product stabilization, **not** a kinetic barrier. Aim 1 explicitly addresses both
  via tighter convergence and external benchmarking — these are budgeted method-validation tasks, not
  assumptions.
- **Model reduction.** The gas-phase reduced core omits protein/phosphate/water; Aim 2's staged
  embedding tests the sensitivity of conclusions to environment.
- **Epitope realism (risk).** Real GADA epitopes are conformational; the current dock uses placeholder
  linear sequences. Aim 3 is gated on obtaining genuine CDR/epitope data and is the highest-risk aim —
  mitigated by anchoring to published GADA epitope maps and by treating Aims 1–2 as independently
  valuable deliverables.
- **Independent benchmarking.** Every RealQM claim used downstream is cross-checked against an
  established quantum-chemistry reference on a system small enough to permit it.

## Broader impact

A validated, GPU-cheap, parameter-free reactive quantum method applicable to cofactor enzymology;
an electronic-structure rationale for GAD65's unique autoantigenicity; and a transferable workflow
linking enzyme cofactor state to immune-epitope presentation — relevant to other PLP enzymes and to
autoantigen biology beyond T1D.
