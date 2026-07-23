# RealQM free-boundary conditions (summary)

Full article: **RealQMAction.tex** ("The Free-Boundary Conditions of RealQM, and their Overdetermination in
Time"). Kept basic: conditions to be satisfied on the boundaries; no domain variation, no Noether.

## Real-valued (static) theory — well posed
N unit-charge densities psi_i^2 (real) on non-overlapping domains, ground state minimises the Coulomb energy
E. Each domain: a one-electron eigenvalue equation with its own Poisson/Hartree potential, coupled by
Coulomb. On the free boundaries TWO conditions hold:
  (i)  continuity of psi (=> density continuity rho_1=rho_2, since rho=psi^2);
  (ii) vanishing normal derivative d psi/dn = 0 (homogeneous Neumann / Bernoulli).
(i) matches the densities; (ii) fixes the interface location. Together the static free-boundary problem is
well posed -- this is the theory that computes atoms/molecules/nuclei. Static, real: density fixed, no
charge flows, the two conditions sit together without tension.

## Time-dependent (complex) extension — OVERDETERMINED
Complex psi (Schrodinger from E) carries a current j_i = Im(psi_i* grad psi_i), d_t rho_i + div j_i = 0
(zero for real psi). Keeping each electron's unit charge as the boundary moves at speed V_n requires a THIRD
condition:
  (iii) V_n = (j_i.n)/rho_i  (boundary moves with the flow; and j_1.n/rho_1 = j_2.n/rho_2).
Three conditions (i)(ii)(iii) on one free boundary => OVERDETERMINED; cannot hold all three.
Concretely: (ii) Neumann => j.n = 0 => (iii) forces V_n = 0 (boundary frozen); and (i) is a condition on the
VALUE of psi while (iii) is on its GRADIENT -- independent.

## The fork (choose which condition to drop)
- (A) Keep psi continuous: retain (i)+(ii) [the static Bernoulli pair], boundary sits where they hold and
  moves as the fields evolve; energy conserved; drop (iii) -> charge MIGRATES between domains (per-electron
  Q_i not conserved, sum is). This is what RealQM's level-set relaxation already does.
- (B) Keep per-electron charge: retain (iii); interface = contact discontinuity (flow speeds match, density
  JUMPS); psi discontinuous, drop (i).
Both reduce to static Bernoulli at rest (currents=0 => V_n=0, boundary stops, (i)+(ii) hold). They differ
only for a genuinely MOVING boundary (real-time multi-domain dynamics). Resolution left OPEN.

## Scope
Only the time-dependent extension is affected. The real-valued static theory (the program's established
results) is untouched: currents vanish, boundary at rest, Bernoulli holds.
