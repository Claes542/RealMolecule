# Yang–Mills vs RealNucleus: A Theory Whose Existence Is Open, and a Model That Runs

**Claim in one line:** the gauge theory meant to explain why nuclei hold together has not yet been shown to exist as mathematics, and it does not compute a nuclear binding energy; a Coulomb model with no strong force exists because it runs, and it computes one. That number can be checked, and it comes out at about 87%.

## Where Yang–Mills came from

In 1954 Chen Ning Yang and Robert Mills wrote down a generalisation of Maxwell's electrodynamics in which the field carries its own charge. The aim was the **strong interaction between protons and neutrons**: isospin made into a local gauge symmetry, with the nuclear force carried by the new field. Pauli raised the objection at once. The quanta of such a field are massless, so the force should reach to infinity, but the nuclear force dies off within a few femtometres.

Twenty years later the theory came back as **Quantum Chromodynamics** (1973), Yang–Mills with gauge group SU(3), acting on quarks. In this version the force between nucleons is no longer fundamental. It is a residue of the colour force, much as the van der Waals force is a residue of electrostatics. Asymptotic freedom earned a Nobel prize, and at collider energies the theory is superb.

## The Millennium Problem: does the theory exist?

In 2000 the Clay Institute listed **Yang–Mills existence and mass gap** among its seven Millennium Problems. Jaffe and Witten's statement asks for a proof that, for any compact simple gauge group, a quantum Yang–Mills theory **exists** on four-dimensional spacetime and has a **mass gap**: a strictly positive energy for the lightest excitation.

Take the two halves separately.

- **Existence.** Nobody has constructed the quantum theory in four dimensions as a mathematical object satisfying the standard axioms. The theory used every day is a set of perturbative rules plus a lattice regularisation whose continuum limit is assumed, not proved.
- **Mass gap.** The gap is what Pauli asked for in 1954. A gap means a short range, and a short range is what makes the nuclear force nuclear. Seventy years on it is still a conjecture, supported by lattice numerics and not proved.

So the property that would make Yang–Mills a theory of short-range nuclear binding is the property that is open.

## What it delivers on nuclei

Grant the theory anyway and ask for the number it was introduced to explain, the binding energy of a nucleus.

- **Lattice QCD** has reached light nuclei only at **unphysical quark masses**, with pions several times heavier than real ones. The extrapolation back to the physical point is not controlled, and different groups have disputed even whether two-nucleon systems bind at those masses.
- **Chiral effective field theory**, the working theory of nuclear forces since Weinberg (1990), is **fitted**. Its low-energy constants are read off nuclear data, and then nuclear data are reproduced.

Seventy years after Yang–Mills and fifty after QCD, there is no parameter-free computation of the deuteron or the alpha from the theory.

## RealNucleus

RealNucleus asks whether the binding can come from the force everyone agrees is there. It has **protons and electrons as charge clouds** on non-overlapping domains, the same free-boundary construction RealQM uses for atoms, interacting by **Coulomb's law only**. There is no strong force and no weak force. A neutron is a proton–electron pair, so a deuteron is two protons held by one electron. Binding comes from **dual confinement**: the electron clouds hold the protons together against their repulsion, and the protons cage the electrons against theirs.

The model is computable. It runs on a GPU in a web browser, so its numbers can be checked against measurement:

- **One scale**, fixed on the deuteron (2.22 MeV). Nothing else is fitted.
- **The alpha comes out at about 87%** of its measured binding, with an alpha/deuteron ratio of **about 11 against 12.7**.
- **The model neutron (1p + 1e) is computed unbound**, so free-neutron decay into a proton and an electron follows from the same Coulomb energy, with no weak force.
- **Binding does not run away under compression.** Scaling the configuration down passes through an interior minimum, so these nuclei have a preferred size chosen by the variational principle rather than imposed.

**A correction, made openly.** Earlier posts and papers in this series gave the ratio as 13.1 and the alpha at 103%. Those were single-mesh numbers. On a four-mesh ladder the deuteron settles about 27% deeper, and the ratio levels off near 11. The heavier rows of the ladder (⁸Be, ¹⁶O) are still **one arrangement selected from a scan**, not mesh-converged, and are provisional. That is the cost of a model that computes: when its numbers are wrong, the error can be found and has to be reported.

## The comparison

| | Yang–Mills / QCD | RealNucleus |
|---|---|---|
| Exists as mathematics | open (Clay problem) | yes: it is an algorithm, and it runs |
| Short range | needs the mass gap, which is unproven | electron glue localised into alpha-sized packets (2e + 4p) |
| Parameter-free nuclear binding | none | alpha ~87%, ratio ~11 vs 12.7, one scale |
| Inputs for nuclei | quark masses, plus fitted constants in practice | proton and electron charge; deuteron fixes the unit |
| What sets nuclear size | the gap (unproven) | open: the model's own unsolved problem |

The last row is where the two meet. Both put the hard question in the same place: **what sets the nuclear scale**. Yang–Mills puts it in a mass gap that has not been shown to exist. RealNucleus finds an interior size minimum but cannot yet predict the size law R ∝ A^(1/3). One of the two can be tested against measurement as it stands.

## The point

This is the distinction drawn in [*The Algorithm Is the Proof*](blog_algorithm_not_theorem.md). A Millennium Problem that asks whether a theory exists is asking for an existence theorem. An algorithm that runs needs no such theorem, because each run exhibits a solution.

Running is not the same as being right. RealNucleus is at 87%, not 100%, and its heavier nuclei are not yet derived. But its errors can be measured, and the measurement has already corrected it once. Seventy years after Yang and Mills, the theory chosen to explain the nucleus still owes both a proof that it exists and a computed nuclear binding energy. That justifies asking how much of the nucleus Coulomb's law already explains.

---

*The model, the mesh ladder and the simulations are in RealNucleus v4 and at [claes542.github.io/RealMolecule](https://claes542.github.io/RealMolecule/gallery.html).*
