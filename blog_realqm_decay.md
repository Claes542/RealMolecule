# Radioactive decay in RealQM: an honest excursion into time-dependent charge densities

Radioactive decay is the textbook poster child of quantum randomness. A nucleus sits there for a
microsecond or ten billion years and then, for no reason anyone can point to, it decays. Standard quantum
mechanics says the moment is *irreducibly* random — uncaused, only its probability defined. So it is a fair
question to put to RealQM, which describes matter not as probability amplitudes but as **charge densities
evolving deterministically in ordinary three-dimensional space**: can a deterministic, real-space theory say
anything sensible about decay?

We spent a long, disciplined excursion finding out. Here is the honest ledger — including, and especially,
the parts that didn't work.

## Two decays, two verdicts

**Alpha decay is the clean case, and RealQM handles it fully.** An alpha particle (a ⁴He nucleus, charge
+2) tunnels out through the daughter's *Coulomb* barrier. It is a genuine two-body decay: no weak force, no
neutrino, and a sharp, *monoenergetic* alpha line whose very sharpness is the proof that no third body is
emitted. Everything the process needs — extended charge, a Coulomb barrier, two-body kinematics — lives
inside RealQM. And there is a genuinely RealQM-specific result underneath it: the binding of the whole
alpha-cluster ladder (⁴He, ¹²C, ¹⁶O, … ⁴⁰Ca) comes out at ~107% of experiment **from Coulomb alone, with no
strong force**, one scale fixed on the deuteron. Alpha decay is where RealQM is at home.

**Beta decay is where the charge-density picture ends — and we say so.** It was tempting to claim beta decay
*without* a neutrino: RealQM conserves energy by construction, so maybe the continuous electron spectrum is
just the conserved energy being partitioned among the electron, the recoil, and the radiated field. We
tested that quantitatively. It fails. The antineutrino carries, on average, about **60% of the released
energy** and the momentum imbalance; the field a charge can radiate is smaller by two orders of magnitude
(the known inner-bremsstrahlung level, ~α). The recoil is negligible. So the neutrino is *not* removed — and
the honest reason is deep: the neutrino is **chargeless**, and a charge-density theory simply has no object
of that kind. Beta decay marks the boundary of the program, and the paper marks it plainly.

## The half-life, three ways — and no WKB

Here is the part that genuinely worked. Textbook alpha lifetimes span **twenty-five orders of magnitude**
(²³²Th at 10¹⁰ years, ²¹²Po at a fraction of a microsecond), and Gamow's 1928 WKB barrier factor famously
reproduces that Geiger–Nuttall law. But WKB is a semiclassical shortcut. Does the *full* time-dependent
RealQM give the half-life directly?

It does. Evolve a metastable charge behind a barrier in **real complex time** (the same solver as the static
relaxation, only the imaginary-time step swapped for a unitary one): the trapped charge decays
**exponentially**, and the half-life is read straight off the dynamics. Sweep the barrier and log t½ stays
linear in √(V−E) — Geiger–Nuttall, from first principles. The narrow, long-lived resonances that real-time
propagation can't reach come exactly from the **complex-energy (Siegert) width**. All three routes agree,
and none uses WKB — which is thereby *validated*, not relied upon. You can watch it happen in the browser:
the charge tunnelling through the barrier while the half-life emerges live.

## Determinism — and the mechanism that died

The most seductive idea was determinism. If RealQM is a deterministic theory, then decay isn't *really*
random — it only looks random because we don't know the exact initial state. That is the century-old
de Broglie–Bohm position, and RealQM carries it naturally: the whole history of a decaying configuration,
tunnelling included, is fixed by its **initial charge configuration**; the apparent randomness of
identical-looking nuclei decaying at different times is *epistemic*, our ignorance of that configuration.

We then reached for something sharper: coexisting charge domains, each carrying a phase clock
e^(−iEₖt/ℏ), with the escape *gated* by the coincidence of their phases — a deterministic mechanism
producing the exponential law as the statistics of a coincidence. It was a lovely picture. **It is also
wrong**, and tracing it to the end is what the excursion was really about.

The refutation is clean. In the full time-dependent RealQM, the escaping domain feels its neighbours *only*
through their **densities** |ψⱼ|², which are phase-invariant; the free boundaries carry **zero flux**. So the
neighbours' phase clocks never reach the escaping domain — the moving free boundary transmits *density, not
phase*. The decay is plain Gamow tunnelling; there is no phase gating. To manufacture gating you would have
to bolt on a **phase-permeable (Josephson) interface** — a thin overlap and a new coupling the variational
free boundary does not give — and it is *unnecessary* anyway, because the density dynamics already carry the
decay and its half-life. So we dropped it. The determinism survives (it's an interpretation); the mechanism
does not.

## So what did the excursion actually net?

No spin: **we did not find new decay physics.** The decay rate is barrier penetration, the same physics
standard quantum mechanics gives. What the full time-dependent RealQM brings, for decay, is *ontological* —
a deterministic, real-space charge-density picture in place of amplitudes and collapse — and *diagnostic*:
it was the tool that let us test and **rule out** the tempting overclaims. The science ended up being in
what we subtracted.

And that is the point worth keeping. Each attractive story — beta without a neutrino, deterministic
phase-coincidence gating — looked good until it was pushed hard, and pushing it turned it into either a
clean negative result or "it's just tunnelling." That is not a failure. It is how you end up with two papers
that claim exactly what is true and nothing more: alpha decay as deterministic Coulomb-barrier tunnelling
with the neutrino nowhere in sight; beta decay honest about the chargeless carrier it cannot supply; the
half-life captured without WKB; and the phase mechanism named, tested, and set aside.

RealQM's real power was never in single-particle escape dynamics — it is in the **static, multi-domain**
world of binding and geometry, where non-overlapping charge domains do genuine work. The one clean theory
question this excursion surfaced is the **correct time evolution of a free boundary** — advection by the
charge-fluid velocity together with a Bernoulli condition — which we identified but did not yet derive.
That, not a new decay law, is the thread worth pulling next.
