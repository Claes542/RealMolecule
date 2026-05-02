# Notes — virial residual as free-boundary convergence diagnostic

Saved for possible later work; was Section 6 paragraph in RealQMarXiv4
(commit 4788609), removed in subsequent commit because the analytical
status of the diagnostic is not yet certain.

---

## Original text (as drafted)

**Virial residual as a free-boundary convergence diagnostic.**

The classical virial theorem $2K + V = 0$, derived from the scaling family
$\psi_\lambda(\mathbf{x}) = \lambda^{3/2}\psi(\lambda \mathbf{x})$ at any
continuum stationary point, holds in RealQM *only at the unconstrained
variational minimum* where the partition $\{\Omega_i\}$ is also at its
variational minimum. The Bernoulli boundary conditions comprise two
requirements at the inter-electron boundary $\Gamma_i$: continuity of
$\psi_i$ across the interface, and the homogeneous Neumann condition
$\partial\psi_i/\partial n = 0$. The first determines where $\psi_i$ and
$\psi_j$ meet; the second is the transversality condition from variation
w.r.t. the boundary location. Both must hold at the unconstrained minimum;
together they fix $\Gamma_i$.

In the molecular case, Slater's virial theorem generalises this:

$$2K + V = -\sum_{a=1}^{M} \mathbf{R}_a \cdot \mathbf{F}_a,$$

where $\mathbf{F}_a = -\nabla_{\mathbf{R}_a} E$ is the force on nucleus $a$.
The residual on the right vanishes only at the equilibrium molecular
geometry. An analogous structure applies in RealQM to the inter-electron
free boundaries: at the constrained variational minimum where continuity
is imposed but the homogeneous-Neumann condition is only approximately
satisfied, the residual $2K + V$ is non-zero and reports how much energy
could be lowered by relaxing the boundary further. The residual functions
as a **convergence diagnostic for the free-boundary structure**, more
informative than the StdQM analogue: in StdQM, $2K + V = 0$ is an
automatic identity at any energy eigenstate (no separate boundary variable
to converge); in RealQM it tracks both the wave-function and the boundary
relaxation, so a simulation that has eigenvalue-converged $\psi_i$ but
boundaries away from joint Bernoulli will correctly fail the virial check,
signalling the residual constraint.

---

## What needs to be checked before re-including

1. The exact form of the virial-residual identity in RealQM with constrained
   boundaries: is it really $2K + V = (\text{boundary residual term})$,
   or is the precise form different?

2. Slater's molecular virial $2K + V = -\sum \mathbf{R}_a \cdot \mathbf{F}_a$:
   verify the sign convention (force = $-\nabla E$ vs $+\nabla E$). My
   derivation in chat conversation gave the opposite sign in one branch,
   so the precise factor needs careful checking against textbook
   references (Slater 1933, or any quantum-chemistry textbook chapter on
   the molecular virial theorem).

3. The "boundary residual" analogue for inter-electron Bernoulli boundaries
   in RealQM: is there a clean integral expression like
   $\sum \int_{\Gamma_i} (\text{something}) \, dS$? Should be derivable
   from the boundary contribution to the variation of $E$ under
   electron-only scaling at fixed boundary, but the explicit form
   wasn't verified.

4. Whether the diagnostic is genuinely useful as a convergence check
   in the simulator: does the residual actually decrease cleanly as the
   simulation converges, or does it have non-trivial finite-size /
   discretisation contributions that swamp the physical content?
