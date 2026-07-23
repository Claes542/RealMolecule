# The equation of motion of the RealQM free boundary (summary)

Full derivation and discussion: **RealQMAction.tex** ("A Stationary-Action Principle for Real Quantum
Mechanics, and the Equation of Motion of the Free Boundary").

RealQM is governed by ONE stationary-action principle whose dynamical variables are the charge-density
fields ψ_i on non-overlapping domains Ω_i(t) AND the free boundaries between them:

    S = ∫dt { Σ_i ∫_{Ω_i} [ (i/2)(ψ_i*∂_tψ_i − c.c.) − ½|∇ψ_i|² ] − ½∬ρρ/|x−y| }.

## Field variation → the time-dependent RealQM equations
    i ∂_t ψ_i = −½ ∇²ψ_i + q_i φ[ρ] ψ_i,   φ[ρ] the self-consistent Coulomb potential.

## Domain variation → the free boundary is a MATERIAL interface
The action is invariant under an INDEPENDENT phase rotation of each domain (ψ_i→e^{iα_i}ψ_i), so by Noether
each electron's charge Q_i=∫_{Ω_i}|ψ_i|² is conserved. With the boundary moving at normal speed V_n,
charge conservation forces, pointwise on the interface Γ,

    V_n = (j_i·n)/|ψ_i|² = v_i·n ,   v_i = ∇S_i  (Madelung velocity),  j_i=Im(ψ_i*∇ψ_i),

i.e. the interface is MATERIAL (no charge crosses) and advects with the charge-fluid velocity; applied to
both domains it forces velocity continuity v_1·n = v_2·n = V_n on Γ.

## Energy by Noether
Time-translation invariance ⇒ total energy E = Σ_i ½∫|∇ψ_i|² + ½∬ρρ/|x−y| conserved EXACTLY for the
coupled field+boundary evolution -- provided the boundary obeys its own Euler-Lagrange law above, not an
imposed rule. Spatial-translation invariance ⇒ momentum conserved.

## Corrects two naive prescriptions
- Dirichlet (ψ_i=0) + slope-matching |∂_nψ_1|=|∂_nψ_2|: conserves per-electron charge but NOT (generally)
  energy -- this is what realqm_twodomain.py used, hence its energy leak.
- Neumann (∂_nψ_i=0) + density continuity |ψ_1|=|ψ_2|: can conserve energy but TRANSFERS charge across the
  moving boundary, violating per-electron charge conservation. (An earlier version of this note wrongly
  favoured it.)
Only the MATERIAL law (boundary moves with the flow) respects BOTH, because both descend from the one
action.

## Static limit & scope
Stationary states have j_i=0 ⇒ V_n=0: the boundary is at rest and S reduces to the static Coulomb-energy
minimisation defining the RealQM ground state (atoms/molecules/nuclei). The time-dependent law matters for
real-time NON-adiabatic multi-domain dynamics (attosecond, charge migration, photochemistry, collisions);
it does NOT affect Born-Oppenheimer chemistry (static re-minimisation per geometry) or radioactive-decay
timing (density-driven, boundary-independent).
