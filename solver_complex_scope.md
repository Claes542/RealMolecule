# Scoping note: a complex / current-bearing extension of the RealQM solver

**Goal.** Test whether the electron's magnetic moment emerges as a *circulating current*
(μ = ½∫ r×J, J = (ħ/m)·Im(ψ*∇ψ)) such that the **free** electron carries ~μ_B (a phase winding
survives) while the **caged** nuclear electron carries ~0 (it drops to the winding-free m=0 state).
Companion to `NuclearMoment.tex`.

## Where the solver is now

`molecule_nucleus.js` (WebGPU) evolves a **real** field ψ by **imaginary-time propagation** (gradient
descent to the ground state). Real ⇒ J ≡ 0 ⇒ no current, no moment. So the current machinery has to be
added. Three levels, cheapest first — and Level 0 needs no new solver at all.

---

## Level 0 — the winding energy-cost integral (existing densities, ~1 hour)

**Key physics that makes this cheap.** A phase winding e^{imφ} adds an azimuthal current whose
**kinetic-energy cost** is
```
ΔT(m) = (ħ²/2m_e) · m² · ∫ |R|² / s²  d³r          (s = cylindrical radius from the winding axis)
```
This cost scales as **1/s²**: tiny for a spread-out free electron, **enormous for a compact caged
electron**. So the mechanism is predicted *without any dynamics*: caging makes the winding so expensive
that the electron prefers m=0 (no moment), while the free electron can afford it (μ_B).

**What to do.** Take the already-converged real amplitudes |R|² the solver outputs for
(a) a free electron and (b) the caged deuteron electron (`nucleus_2h.html` / the flat-electron runs),
and evaluate ΔT(1) for each. Prediction to confirm:
- free electron: ΔT(1) ≈ order of the atomic binding (winding affordable → μ_B survives);
- caged electron: ΔT(1) ≫ binding, femtometre-scale (winding unaffordable → m=0 → μ = 0).

**Effort.** A post-processing reduction over the existing density grid (JS or a 20-line Python read of
the exported slice). No kernel changes. This alone substantiates the paper's central claim
energetically.

---

## Level 1 — constrained complex minimization (moderate, ~half a day)

Make ψ complex (`Re`, `Im` fields), **fix the winding number m** as a topological boundary condition
(phase = m·atan2(y−y0, x−x0) about the electron-domain axis), and relax the amplitude R by the existing
imaginary-time step applied to |ψ|. Then compare total energies E(m=0) vs E(m=1) for free vs caged.

**Code changes to `molecule_nucleus.js`:**
- Store ψ as two real fields (or `vec2` per cell); the Coulomb potential couples only to |ψ|² = Re²+Im².
- Kinetic term uses the full complex Laplacian; the imposed phase enters through the winding factor.
- Reuse the existing free-boundary and Coulomb machinery unchanged.

**Answers:** whether m=1 is energetically favored (free) or disfavored (caged) self-consistently, i.e. a
first-principles version of Level 0.

---

## Level 2 — real-time complex propagation (the full test, ~1–2 days)

Replace imaginary time with **real-time Schrödinger** so a current can form and be watched:
```
iħ ∂ψ/∂t = Hψ ,   H = −c ∇² + V ,   c = ħ²/2m_e
 ⇒  ∂Re/∂t = +(1/ħ) H Im ,   ∂Im/∂t = −(1/ħ) H Re
```
Use **Visscher staggered leapfrog** (Re and Im offset by ½ dt) — cheap, norm-preserving to O(dt²),
stable for dt < ħ/H_max (same CFL family as the current step).

**Code changes:**
- Two coupled update kernels (Re, Im) replacing the single real ITP kernel; Laplacian and V reused.
- **Initialize** ψ = R·e^{iφ} with R the converged real ground state and φ the winding about the domain axis.
- **Observables** each frame: current J = (ħ/m)(Re∇Im − Im∇Re); moment μ_z = (−e/2)∫(x J_y − y J_x) dV
  (GPU reduction or slice read-back), reported in μ_B.

**Test protocol:**
1. Free electron: init with m=1 winding, evolve, read μ_z → expect it to hold near −μ_B (winding stable).
2. Caged deuteron electron: init with m=1 winding, evolve, watch whether the winding **unwinds** (μ_z → 0)
   as the compact geometry makes it energetically untenable. That unwinding, if it occurs, is the direct
   dynamical demonstration: free ⇒ μ_B, caged ⇒ 0.
3. Optional: seed a **structured** (core+/halo−) current and see whether a small, non-quantized residual
   (the μ_N-scale neutron value) is dynamically stable.

**Answers:** the full self-consistent free-vs-caged moment, dynamically, and whether a small residual
survives — i.e. the pieces `NuclearMoment.tex` lists as open.

---

## Recommended path

Do **Level 0 first** — it is nearly free, uses data already in hand, and if ΔT(1) comes out
tiny-for-free / huge-for-caged it already backs the paper's claim quantitatively. Escalate to Level 1/2
only to turn the energetics into a self-consistent, dynamical demonstration for a stronger version of the
article. Level 2 is also the natural vehicle for the free electron's own μ_B (the winding as the
Zitterbewegung-style internal circulation) — the remaining hard piece, short of the g-factor anomaly.
