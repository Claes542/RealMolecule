

# Response to the Reviewers

**Manuscript** 1568909 — *A 3D Multiphase Continuum Model for Atoms*
**Journal** International Journal of Quantum Chemistry
**Decision** Major revision

---

We thank both reviewers for reports that were detailed, specific, and — on the points that
mattered most — correct. Reviewer 2's central objection and Reviewer 1's "elephant in the
room" both identified places where the manuscript's framing claimed more than its body text
supported. We have not argued with either. The revision resolves them by **removing the
material that could not be defended** and by **promoting to the foreground the result that
Reviewer 2's objection actually points at**.

The revised manuscript is substantially shorter, and its central claim has changed. We set out
the scope reduction first, then the change of framing, because between them they account for
most of what a reviewer will notice.

## The principal change: scope reduced, parameters eliminated

The original manuscript reported the main group from Li to Rn. Those heavy-element results
were obtained with a frozen core and a pseudo-kernel carrying a softening radius $r_c$,
calibrated against one reference atom per period — the very construction Reviewer 2 identified
as contradicting the abstract's "in-principle parameter-free" claim.

We considered two ways to answer this: restate the claim honestly while keeping the tables, or
remove the construction. **We have removed it.** The frozen core, the pseudo-kernel, the
two-scale core/valence decomposition, and both heavy-element tables are gone from the paper.

The consequence is a genuine trade, and we state it plainly rather than presenting it as a
pure improvement:

- **Lost:** the reach from argon to radon, and the two tables reporting it.
- **Gained:** every number now in the paper is computed all-electron, with the bare nuclear
  term $-Z/r$, with no core radius, no screened kernel and **no fitted length anywhere**. The
  claim that nothing is adjusted is now literally true of every result reported, which it was
  not before.

The paper now covers periods 1–3, hydrogen through argon, and says so in the title matter,
abstract, introduction and conclusion. We think a smaller paper that is parameter-free
throughout is worth more than a larger one whose central claim a reader has to qualify — and
Reviewer 2's report is what convinced us of that.

One limit at that boundary is physical and one is not, and the difference matters. The physical
one: a build-up that adds outward cannot reach a shell lying inside one already occupied, which
is what the fourth row requires, so the configurations there are supplied rather than derived.
The other was numerical. The innermost shell contracts as 1/Z - we measure r_1s * Z = 4.3,
constant to +-9% from carbon to argon - so at fixed mesh it is eventually spanned by too few
cells and the energies degrade, going positive around iron. Refining the mesh removes it
entirely: at N=800 the all-electron total energies run from argon to zinc within 1-5%, with no
trend in Z and nothing fitted. We had taken that degradation for a limit of the model; it was a
limit of the grid.

## A change of framing, prompted by R2.2

**The algorithm is a relaxation, not a global minimisation.** None of the three updates
searches the space of configurations; each is a parabolic relaxation, and what they reach
together is a *steady state*. The relaxation arrives at whichever equilibrium lies at the end
of the path it is on, and at no step is a lower configuration elsewhere offered and refused.
**Energy minimisation here is path dependent.** Ordinarily that is invisible, because the
reachable equilibrium *is* the minimum. It is the whole point in the Aufbau, where a lower
configuration exists, is never reached, and is not the physical one either.

Three figures have been added, none of which was in the original.

---

# Reviewer 1

**R1.1 — The ontological status of RealQM. The justification is deferred to a reference that is
itself under review, and the claim that RealQM is a correction to rather than an approximation
of standard quantum mechanics cannot rest on atomic results alone. At minimum the ontological
claims should be phrased as speculation.**

We accept this without reservation and have taken the off-ramp the reviewer offered. Section 1
now closes with a paragraph (*Comparison with textbook quantum mechanics*) stating that the
two are different mathematical objects — a linear Schrödinger equation for one wave function on
$3N$-dimensional configuration space, versus a nonlinear free-boundary problem for $N$ charge
densities on three-dimensional physical space — that exclusion is realised differently in each,
and that **neither is shown here to be an approximation to the other**. It states explicitly
that the wider ontological reading advanced elsewhere should be taken as speculative pending
evidence stronger than atomic energetics can supply, and that nothing in this paper depends on
it. What is compared here is what each produces for the periodic table, and nothing more.

**R1.2 — Could equivalence with density functional theory be established?**

Not attempted, and we should say why we do not expect it to be a short step. RealQM is
fundamentally different from DFT in the object it works with. DFT is built on a single common
density rho = sum_i |psi_i|^2, and Hohenberg-Kohn is a statement about that one function. RealQM
has no such object in that role: its state is N *individual* charge densities together with the
partition of space between them, which carries strictly more information than their sum. Two
different partitions of the same total density have different energies, because the i != j rule
counts pairs over disjoint domains rather than integrating a common rho against itself.

The consequence is visible. DFT's Hartree term integrates rho against itself and so carries a
spurious self-interaction, which the exchange-correlation functional is partly there to remove;
here there is nothing to remove. An equivalence would have to relate objects of different kinds,
and relate an approximated functional to one written down exactly. We name it as open.

**R1.3 — Does the method scale to molecules? Is the spherical model useful beyond atoms?**

The spherical reduction is specific to atoms and does not extend; it is available here only
because an atom is spherically symmetric, and the revised introduction now derives the
reduction from that fact rather than presenting it as an implementation choice. The
three-dimensional form of the model carries no such restriction and is the one that extends.

On molecules, what the method supplies first is **forces**. The free boundaries move with the
charge, so the force on each kernel is available at every step of the same relaxation, with no
separate apparatus and no derivative of a fitted functional; geometry follows from it, and
molecular dynamics follows without further machinery. Energy differences within one composition
are reliable for the same reason, the discretisation error being common to the states compared.
Absolute energetics across compositions is the harder case and we make no claim about it here.
No molecular result is reported in this paper.

**R1.4 — Could basis sets reduce the cost, particularly for periodic systems?**

We do not think a basis is needed; the mesh is enough. A basis exists to make the integrals of
a configuration-space method tractable and to compress the representation, and neither problem
arises: there are no two-electron integrals, the interaction coming from a Poisson solve on the
same grid, and the cost is already linear in the number of electrons.

There is also a reason of principle. The unknowns include the interfaces, which are geometric: a
domain boundary is a moving surface, represented directly by a mesh, where a global basis would
have to build a nearly discontinuous partition out of smooth functions. And a basis is an
empirical input whose adequacy must be established per system; introducing one would give back
exactly the parameter-free character R2.1 asks us to secure. Where more accuracy is wanted we
refine the mesh.

**R1.5 — Is the imaginary-time propagation connected to path-integral or centroid formalisms?**

No. It is a relaxation scheme used to reach the stationary state and has no such connection.
Only the converged solution enters any result, so the temporal ordering of the relaxation is
immaterial. This is now stated in §Limitations.

---

# Reviewer 2

**R2.1 — The abstract claims an in-principle parameter-free calculation, but the results use
per-period $r_c$ calibrations, and separate ones for total energies and ionization energies.**

Correct, and this was the decisive report on the paper. The two statements in the original
manuscript could not both stand. As described above, we did not repair the claim — we removed
the construction that violated it. There is no $r_c$ anywhere in the revised paper, no
calibration of any kind, and no per-element or per-period empirical input. The inputs are the
nuclear charge and the number of electrons.

What made that possible is worth reporting, because it was not obvious to us. The calibrated
radius existed to let the heavier elements be computed on a fixed mesh; its real function was
to remove the innermost shell, which is the one a uniform grid resolves worst. But that is a
resolution problem and it has a resolution answer. The free-boundary shell radii are outputs of
the model, and we now measure the innermost one to follow $r_{1s}Z \approx 4.3$, constant to
$\pm9\%$ from carbon to argon. Refining the mesh accordingly — and scaling the step count with
it, since the relaxation is explicit — the all-electron energies run from argon to zinc within
1–5% with nothing fitted. The parameter was standing in for a mesh.

(The original abstract also asserted a core radius scaling as $1/Z$ as a *result*. The body
never supported it and it has been withdrawn; what is stated above is a measurement used to
choose a grid, not a claim about the model's output.)

**R2.2 — The radial model favours cap-8 packing, yet cap-4 configurations are chosen because
they agree better with reference energies.**

The reviewer's reading was accurate: the packing was selected by fit. What the revision adds is
a measurement of why, and the answer is structural rather than a careless search.

**The energy is monotone in the packing.** There is no interior minimum, so the unconstrained
minimiser always absorbs the core into the valence. Beryllium: a single 4-shell at -18.893 Ha,
28.8% too deep, against the physical 2+2 at -14.607 Ha, 0.41%. Wrong in kind, not in accuracy,
and worsening with Z.

**But it fails only where it may dissolve a formed shell.** Among configurations that keep one
it selects correctly - Be's 2+2 over 2+1+1 by 0.13 eV. And this is not a constraint we impose.
The algorithm is a relaxation, so it reaches the equilibrium at the end of its path; a filled
shell is more rigid than the charge arriving at it, and the interface moves only where one
domain's amplitude locally exceeds the other's. The newcomer is turned aside because it cannot
win that comparison, not because a rule forbids it. What decides is *relative* rigidity, and
nothing else could: the front is a difference of amplitudes.

**Figures 2 and 3 show it happening.** Li+ with one electron added, three-dimensional,
all-electron, no r_c, the core an ordinary deformable domain. The charge is launched as a
half-shell, with no shell present at the start; it wraps around the core and closes into one,
its centroid falling from 0.84 to 0.014 a.u., while the core holds its two charges unchanged.
The merged single-shell state was reachable throughout and lies 1.055 Ha *lower*. It is
declined. Path dependence measured rather than argued.

We report the rule as established for Li through C and **undetermined from nitrogen onward**,
where a shell holding two charges may be a closed pair or a half-filled tetrahedron and the
model cannot distinguish them.

**The capacities are not fitted either.** The two basic filled packings are established by the
calculations themselves: the antipodal pair, 2, and the dual tetrahedra, 8 = 4+4, the latter
against a single 8-shell that over-binds by 9-22%. Composing rows from those two units alone
gives the row lengths 2, 8, 8, 18, 18 and the closures 2, 10, 18, 36, 54 - every noble gas
through xenon - with the fourth row as 2+8+8 rather than 9+9. Nothing is imported and no tile is
needed that the packing cannot build. The account runs out at the sixth row, where 32 is not
2+8+8 repeated, and we report it as exact through xenon and open beyond.

We should be explicit about what we are *not* claiming. The familiar 2n^2 also reproduces the
sequence and reaches radon, but n^2 = sum(2l+1) is the hydrogenic degeneracy doubled by spin, so
offering it as this model's derivation would borrow the answer from the theory being compared
against - and it requires 3x3 and 4x4 tilings the packing has no way to assemble. The earlier
draft of this response rested the scope boundary on that impossibility. We no longer do: under
the packing account the vocabulary is not exhausted at argon, and the boundary rests on the two
limits we can demonstrate - a build-up that adds outward cannot reach an inner shell, and the 1s
contracts as 1/Z so a fixed grid stops resolving it.

**R2.3 — Equation (1) identifies $-\varepsilon_i$ with the optimised neutral−ion energy
difference. Why should these be equal?**

They are not, and we measured the discrepancy before deciding what to do about it. For
bare-nucleus balanced configurations, Koopmans $-\varepsilon_i$ against $\Delta$SCF against
NIST (eV):

| | Koopmans | $\Delta$SCF | NIST |
|---|---:|---:|---:|
| C | 4.85 | 8.74 | 11.26 |
| O | 6.79 | 12.04 | 13.62 |

The frozen estimate is roughly half the relaxed one, so the identification in the original
Eq. (1) was unsupported. Worse for the method, $\Delta$SCF cannot simply replace it: it
requires choosing the cation's packing, and for nitrogen neither choice works — repacking gives
$+51.4$ eV, decrementing the outer shell gives a state *below* the neutral and a negative
ionization energy. **$\Delta$SCF therefore inherits the packing-selection problem of R2.2.**

Given this, we concluded that ionization is the wrong observable for a ground-state model:
it is a *process*, in which the remaining electrons repack, and the model has no principled way
to select the ion's packing. **All ionization-energy material has been removed from the
paper**, and the abstract and Scope now state that ionization and excitation are left aside,
with that reason given.

In its place we report a quantity the model is built to answer: the **Allen configuration
energy** $\chi = (n_s\varepsilon_s + n_p\varepsilon_p)/(n_s+n_p)$, computed natively from the
shell energies. It is a state property of the neutral atom — no ion, no anion, no relaxation,
no calibration — and it is what bond polarity is built from. The **ordering across period 2 is
exact**. The gradient is too steep by a factor $2.23$ (span $51.1$ against $22.9$ eV), with the
error concentrated at the electronegative end and climbing monotonically from $1.09$ at Be to
$2.06$ at Ne. We present the result where it is strong and state the rest as the known
light-$p$-block limitation, whose structural cause (the outer tetrahedron placed at too large a
radius in a radial model) and cure (the three-dimensional solve) are both identified.

**R2.4 — How were the transition-metal points of Figure 1 obtained, and do they establish the
claimed 4s-before-3d removal order?**

They do not. On checking the generating code we found the $s^2$ shell is placed **outermost by
construction**; the run then tests whether the outermost shell is the $s$ shell, which in a
nested radial model the input very nearly guarantees. The reviewer's suspicion was correct:
4s-before-3d was an input, not a result. Two further defects: those runs used 4000 relaxation
steps, which is below this project's own convergence floor, and their total energies were
floor-carried and not independently meaningful.

**The claim has been removed from the abstract and from the paper.** The $d$-block is now
outside the stated scope entirely, and §*Boundary of the spherical model* reports the $4s/3d$
interpenetration as the structure the radial reduction cannot represent — described as a
limitation, with no ordering claimed as a result. §Limitations states that the placement of the
$d$-shell inside the $s$-shell is imposed, not derived.

**R2.5 — The Kr→Xe increase contradicts the claimed down-group trend.**
**R2.6 — Argon appears as 11.29 eV in Table 3 and 8.0 eV in Table 5.**
**R2.7 — $p$-block ionization energies repeat across periods 2–4.**

These three concerned ionization energies of heavy elements and are resolved by the removals
described under R2.1 and R2.3: both source tables are gone from the paper, as is all
ionization-energy material. We record the diagnosis rather than leave it unstated, since each
was a real defect and the reviewer was right about all three:

- **R2.6** was the two calibrations of R2.1 surfacing — the same atom appearing under the
  total-energy calibration and the ionization-energy calibration without either being labelled.
- **R2.7** was structural, not a transcription error. With the core absorbed into $r_c$ and the
  screened charge identical for every noble-gas valence, each column member presented the *same*
  valence problem, so the entries repeated by construction: the model reproduced the column and
  had no representation of the period. That is a serious limitation of the frozen-core
  construction and is among the reasons we removed it rather than defended it.
- **R2.5** we did not fully diagnose before the construction that produced it was withdrawn.

**R2.8 — Grid convergence is demonstrated only for the helium total energy.**

The revised paper reports the mesh ladder it actually has, and does not generalise from it. A word on what
convergence means for the three-dimensional runs, since the revision adds three figures drawn
from them. In the present setting --- the
simplest possible implementation, on the smallest systems --- those runs display *qualitative*
features of the solution rather than precise energies or geometry: which charge holds which
region, where an interface stands, whether an arriving charge closes into a shell rather than
merging inward. What is asked of refinement there is that these features do not change under
it. The precise numbers in the paper come from the radial solver, where the ladder applies. It
also now uses that ladder to make a substantive point rather than only a numerical one.

Helium is the cleanest available test of the spherical reduction — one shell, two electrons, no
core, no $r_c$, and an exact reference — and the revision adds an analysis of what the reduction
costs, which we believe answers the deeper form of the reviewer's question. The
$(N_s-1)/N_s$ factor applied to the same-shell term is not a correction added to the model: it
is forced bookkeeping, removing exactly the $N_s$ spurious self-pairs that smearing $N_s$
one-electron domains into one spherical density introduces. But smearing does a second thing,
and that one is a real loss: it replaces the actual pair *separations* by their spherical
average. For unit charges at a common radius $R$ (units of $1/R$):

| $N_s$ | homogenised, $q=N_s$ | $\times (N_s-1)/N_s$ | packed | ratio |
|---:|---:|---:|---:|---:|
| 2 | 2.000 | 1.000 | 0.500 (antipodal) | 2.00 |
| 3 | 4.500 | 3.000 | 1.732 (trigonal) | 1.73 |
| 4 | 8.000 | 6.000 | 3.674 (tetrahedral) | 1.63 |

The reduction has the pair *count* right and the pair *geometry* wrong, overestimating
intra-shell repulsion by a factor of 1.6–2. Most of that is absorbed by the shell radius, which
is free; what survives appears as systematic under-binding, largest where the ratio is largest.
Helium, at $N_s=2$, under-binds by $2.0\%$ — the sign the table predicts. What the reduction
discards is precisely the angular correlation, which is what packing *means*, and this is why
the radial solver cannot adjudicate a packing question on energetic grounds and why the
capacities are established by the counting argument of R2.2 rather than by radial energies.

We have added this as a stated limitation of the method rather than waiting to be asked.

---

# Summary of changes

**Removed**
- All ionization-energy material, including Eq. (1)'s Koopmans identification, and the tables
  underlying R2.5–R2.8.
- The frozen core, the pseudo-kernel, $r_c$, and the two-scale core/valence decomposition.
- Both heavy-element tables (representative frozen-core atoms; full main group Li→Rn).
- The "4s before 3d" claim (R2.4) and the "in-principle parameter-free" claim as originally
  worded (R2.1) — the latter now restated and true.
- The sentence Reviewer 2 quoted on the correctness of the radial description.

**Added**
- §*Aufbau: how the shell structure forms* — the build-up as the constraint that makes energy
  minimisation select correctly, with the monotonicity measurement (R2.2).
- The capacities composed from the model's own packings (the antipodal pair and the $4{+}4$
  octet), giving row lengths $2,8,8,18,18$ and the closures through xenon, with $2n^2$ set
  aside as the hydrogenic count rather than claimed as a result.
- §*Electronegativity from the shell energies* — the Allen configuration energy, replacing
  ionization as the valence observable (R2.3).
- An analysis of what the spherical reduction costs, with the intra-shell repulsion table
  (R2.8).
- The speculation caveat on ontological claims (R1.1), the molecular limits (R1.3), the basis-set
  concession (R1.4), and the imaginary-time clarification (R1.5).

**Restated**
- Scope: periods 1–3, hydrogen through argon, all-electron, parameter-free throughout.
- The shell structure as an output of the build-up, not of energy minimisation *only* --- the
  energy selects correctly among the configurations the build-up leaves available.
- The Aufbau result as established for Li–C and open from nitrogen.

We are grateful for both reports. The revision **retracts** several claims the original made and
we have tried to save none of them: the parameter-free assertion as it was worded, the
4s-before-3d ordering, the ionization energies, the $1/Z$ core radius, and the reach beyond
argon. What replaces them is new, and follows from the objections rather than working around
them: the Aufbau as a path-dependent relaxation rather than a fitted choice of packing, with the
build-up shown directly in three dimensions; the shell capacities derived from a counting
argument instead of selected by agreement; electronegativity computed natively from the shell
energies, in place of the ionization energies withdrawn; and a stated account of what the
spherical reduction costs, which no one asked for and which the method needed.

So the paper claims less in the places the reviewers challenged, and more, on firmer ground, in
the places the challenges led us to. We think it is a considerably better paper for it.
