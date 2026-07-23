# Radioactive decay in RealQM: how far does deterministic charge take you?

**Claim in one line:** the half-life of a decaying nucleus, and the clean two-body case of alpha decay, come straight out of charge tunnelling a Coulomb barrier — deterministically, with no dice and no WKB shortcut; and the two tempting extras — beta decay *without* a neutrino, and a "phase-coincidence" trigger for the timing — both fail, which is the honest half of the story.

## The starting point, and the problem

Radioactive decay is the textbook poster child of quantum randomness. A nucleus sits there for a microsecond or ten billion years and then, for no reason anyone can point to, it decays. Standard quantum mechanics calls the moment *irreducibly* random — uncaused, only its probability defined.

RealQM describes matter differently: not as probability amplitudes but as **charge densities evolving deterministically in ordinary three-dimensional space**. So it is a fair question to put to it — can a deterministic, real-space theory say anything sensible about decay? We chased that question to the end. Here is the ledger, including, and especially, the parts that didn't work.

## Alpha decay is the clean case — and RealQM owns it

An alpha particle — a helium-4 nucleus, charge +2 — tunnels out through the daughter's **Coulomb** barrier. It is a genuine *two-body* decay: no weak force, no neutrino, and a sharp, *monoenergetic* alpha line whose very sharpness is the proof that no third body leaves. Everything the process needs — extended charge, a Coulomb barrier, two-body kinematics — lives inside RealQM.

And underneath sits a genuinely RealQM result: the binding of the whole alpha-cluster ladder — helium-4, carbon-12, oxygen-16, on up to calcium-40 — comes out at about **107% of experiment from Coulomb alone**, no strong force, one scale fixed on the deuteron. Alpha decay is where the charge-density picture is at home.

## The half-life, three ways — and no WKB

Textbook alpha lifetimes span **twenty-five orders of magnitude**, from thorium-232 at ten billion years to polonium-212 at a fraction of a microsecond. Gamow's 1928 barrier formula famously reproduces that Geiger–Nuttall law — but it is a semiclassical shortcut. Does the *full* time-dependent RealQM give the half-life directly?

It does, and three independent ways agree. Evolve a trapped charge behind a barrier in **real time** (the same solver as the static one, only the time step made unitary): the trapped charge decays *exponentially*, and the half-life reads straight off the dynamics. Sweep the barrier and the Geiger–Nuttall line falls out. The very long-lived cases, too slow to watch, come exactly from the **complex-energy resonance width**. None of it uses WKB — which is thereby *validated*, not leaned on. You can watch it live in the browser: charge tunnelling through the barrier while the half-life builds up on a log plot.

## Determinism: everything hangs on the initial state

Here is the interesting part. If RealQM is a deterministic theory, then decay isn't *really* random — it only looks random because we don't know the exact starting configuration. That is the century-old **de Broglie–Bohm** position, and RealQM carries it naturally: the entire history of a decaying nucleus, tunnelling and all, is fixed by its **initial charge configuration**. Identical-looking nuclei decay at different times because they are not actually identical — they differ in details we neither see nor control. The randomness is *epistemic*, ours, not the world's.

## The mechanism that failed — stated plainly

The seductive next step was to name a *mechanism* for that determinism: let the coexisting charge domains each carry a phase clock, and let the escape be **gated** by the coincidence of those clocks, producing the exponential law as the statistics of a coincidence. A lovely picture. **It is wrong**, and tracing it to the end is what the whole excursion was really about.

The reason is clean. In the full theory the escaping domain feels its neighbours *only* through their charge **densities**, which don't depend on phase; the boundaries between domains carry no flux. So the neighbours' phase clocks never reach the escaping charge — the moving free boundary transmits *density, not phase*. The decay is plain tunnelling; there is no phase gating. To manufacture one you would have to bolt on an extra coupling across the boundary that the theory does not give — and it would be *pointless* anyway, since the density dynamics already deliver the decay and its half-life. So we dropped it.

## Beta decay is the other wall — the neutrino it cannot supply

It was equally tempting to claim beta decay with **no neutrino**: RealQM conserves energy automatically, so maybe the continuous electron spectrum is just the conserved energy shared among the electron, the recoil, and the radiated field. We checked the numbers. It fails. The antineutrino carries, on average, about **60%** of the energy and the momentum imbalance; the field a charge can radiate is smaller by two orders of magnitude. The deep reason is simple: the neutrino is **chargeless**, and a charge-density theory has no object of that kind. Beta decay marks the boundary of the program, and we mark it plainly.

## What it means

So decay, read through RealQM, splits cleanly. The **half-life and the clean alpha case** are charge tunnelling a Coulomb barrier — deterministic, real-space, reproducing the twenty-five-decade Geiger–Nuttall law without WKB, with the timing fixed by the initial configuration in the manner of de Broglie and Bohm. The **two extras** — beta without a neutrino, and a phase-coincidence trigger — both fail, one on the chargeless carrier it cannot supply, the other because a moving boundary carries density and not phase.

No spin on the net result: for decay, RealQM does not find *new* physics. The rate is barrier penetration, the same physics standard theory gives. What the deterministic charge-density picture brings is *ontological* — a real-space account with no dice and no collapse — and *diagnostic*: it was the tool that let us test the tempting stories and **rule them out**. The science ended up being in what we subtracted. That is how you get papers that claim exactly what is true and nothing more — and it is worth more than one more mechanism that doesn't survive contact with the arithmetic.
