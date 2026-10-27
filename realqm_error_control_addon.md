# Proposed add-on: output error control for RealQM

**Status.** A proposal, not a result. The reference implementation is
`tools/aposteriori_bracket.py`, verified on radial hydrogen; nothing has been added to any solver.

---

## What is missing

The RealQM codes report **iteration tolerances** — `tol=1e-11` in 29 places, `tol=1e-12` in 16,
`CONVERGED at step N` in 6 — and otherwise watch for **"drift"** (19 occurrences). A tolerance says
the fixed point stopped moving. It says nothing about the distance from the computed number to the
true one, and the two are not related: on 2026-10-26 a lithium calculation reported constraint
satisfaction at **4×10⁻¹²** while being **64% wrong**.

The same repository already rejects this practice in print. `clay_navier_stokes.tex`:

> "the error in the output is not estimated by comparing successive meshes and hoping, but bounded by
> a quantity the computation itself produces. Adaptivity and error control are the same mechanism."

That machinery appears in ten fluid-side documents and in **no** RealQM solver, no RealQM tool, and
nowhere in the book's 405 pages.

## What can be picked up

Dusson & Maday, *An overview of a posteriori error estimation and post-processing methods for
nonlinear eigenvalue problems* (arXiv:2303.01273), covers exactly this problem class — nonlinear
eigenvalue problems arising in electronic-structure ground states. Three mechanisms, in increasing
cost.

### (1) The conforming upper bound — no extra computation

For a conforming discretisation, `E(u_δ) ≥ E(u)`: the number already computed bounds the model's own
ground state from above.

**Not currently available**, because "conforming" demands three things the codes don't do:

| requirement | what the codes do now |
|---|---|
| the state must be a genuine H¹ function | they hold a grid vector; its P1 interpolant qualifies |
| the constraints must hold for *that function* | `∫u²=1` is imposed as a grid sum |
| the energy must be the continuous functional of it | the energy is a grid sum |

**To add:** interpolate the converged state, re-normalise against the exact mass matrix, re-evaluate
the energy with quadrature. One pass over the solution, no solve.

**What it buys:** the separation the codes currently lack — distance to *RealQM's own minimum* is
numerical error; distance from that minimum to measurement is model error. Without it, a result like
−12.28 against lithium's −7.478 cannot be attributed to either.

### (2) The post-processing and the error estimate — one linear solve

Their (2.7): on a refined space, orthogonal to the computed state,

```
(A + 2 g'(u²)u² + g(u²) − λ) w  =  −(A + g(u²) − λ) u_δ
```

The right-hand side is the residual of the eigenvalue equation; the operator on the left is the
Hessian. Then `ũ = u_δ + w`, and

```
error  ≈  E(ũ) − E(u_δ).
```

**Verified** (`tools/aposteriori_bracket.py`, radial hydrogen, exact −0.5, nothing exact used):

```
  n   k     E(u_d)      estimate    truth   captured   err of E(u~)
 40   4  -0.40530875   -0.08270   -0.09469     87%      1.20e-02
160   4  -0.48858060   -0.01062   -0.01142     93%      7.95e-04
160   8  -0.48858060   -0.01122   -0.01142     98%      2.00e-04
320   8  -0.49688808   -0.00306   -0.00311     98%      5.05e-05
```

So one solve returns the error to **2%** and an energy **~57× better**. The paper notes the
post-processing doubles the convergence rate of both the state and the energy, which is what the last
column shows.

**For RealQM the mapping is direct:** `A` is the localisation operator, `g(u²)` the Hartree potential
of the *other* domains (the i≠j Coulomb term the codes already build), `λ` the computed multiplier.
The only new object is the response term `2g'(u²)u²` — the derivative of the Hartree coupling — plus
a refined space to solve on.

### (3) The asymptotic lower bound — not reached

Their (2.6), `E(u) ≥ E(u_δ) − ½a(w,w)`, **held in no row** of the test. It is asymptotic, and errors
of 1–10% are not asymptotic: P1 elements on a uniform mesh converge slowly against the 1/r cusp.
Graded meshes would be needed. Recorded as not achieved rather than tuned into apparent success.

## Where it would go

- the radial atom solver (`atom_simulator.html` / `tools/atom_headless.js`) — cheapest first target,
  one dimension, and the place where the published total energies come from;
- the cell driver behind the alkali series (`real_plasma_numerics/metal_alkalis.py`);
- `molecule.js` last — the 3D case, where a refined space is expensive and the Hessian solve is the
  real work.

## What it does **not** cover

All of it holds **at a fixed partition**. RealQM's free boundary is an extra geometric unknown and
the Taylor expansions behind both estimates assume smooth dependence on the state. So this controls
the energy *given* the interface and says nothing about where the interface belongs — which is
precisely where the 2026-10-26 failures lived: an interface at 0.446 against the established 2.00,
and optima reported from scan bounds.

And one sharper limit. Because the upper bound holds for any admissible partition, a lower conforming
energy would mean a partition closer to the model's minimum — a rigorous criterion for comparing
partitions. **But that is only valid under the formulation in which Γ is fixed by minimising the
energy.** Under the theory's own formulation — Γ fixed by requiring the domain solutions to match —
lower energy is *not* the test, and a correct partition may have higher energy than some other
admissible one. So the add-on certifies convergence at a given partition and cannot adjudicate
between partitions.

## It is not engineering. It is the existence theory.

An earlier version of this document filed this as the one open item that was engineering rather than
physics. That was wrong, and the reason matters.

**Pure existence says nothing.** Three existence results are in play as of October 2026 and not one
yields a number about matter. A singularity exists for Navier--Stokes under some force -- which says
nothing about the drag on a wing. QCD exists as a four-dimensional quantum field theory -- which says
nothing about why helium-4 binds. A RealQM minimiser exists once a perimeter term is included, by BV
compactness -- which says nothing about the interface, the geometry, or the energy. Each theorem is
real and its information content about matter is zero.

**Constructive existence says everything, and this repo already defines the standard.**
`clay_navier_stokes.tex`: the error in the output is "bounded by a quantity the computation itself
produces", and "adaptivity and error control are the same mechanism" -- with the four-colour theorem
cited as the precedent for computation accepted as proof, *because* it was reproducible and its error
controlled. Not "a solution exists" but "here is one, and here is how far it is from the truth".

**So the bound is not hygiene -- it IS RealQM's existence theory**, in the only sense that carries
information. Without it a calculation asserts a number. With it, every calculation carries a theorem:
the conforming upper bound `E_computed >= E_min` is a rigorous statement about the model's minimum,
available for one pass over the solution and no solve.

**And it levels the two programmes, which are not currently level.** The fluid side meets this
standard: outputs computed, error bounded by the dual, adaptivity and error control unified. The
quantum side reports `tol=1e-12` in 29 places and watches for "drift" in 19. That asymmetry has been
the odd feature of this repository all along, and it now has a name -- the fluid work is constructive
in the author's own sense and the quantum work is not yet.

**What is reached and what is not.** The upper bound is rigorous and cheap. The post-processing
estimate is asymptotically exact but not a bound (98% of the error on the hydrogen test). The
asymptotic lower bound held in no row. So RealQM's constructive existence would at present be
**one-sided**, where G2's is two-sided because the dual supplies the other half. That gap is the real
remaining work, and it is mathematics rather than coding.
