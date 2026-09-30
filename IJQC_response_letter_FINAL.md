

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
28.8% too deep, against the physical 2+1+1 at -14.603 Ha, 0.44%. Wrong in kind, not in accuracy,
and worsening with Z.

**But it fails only where it may dissolve a formed shell.** Among configurations that keep one,
the structure is set by the balanced condition rather than by the energy. For beryllium the row's
capacity is 8, two cap-4 shells, and the period's two charges take one place in each: 2+1+1. The
energy does prefer 2+2, and by a converged margin: on a step ladder flat from 16,000 to 64,000
steps, 2+2 gives -14.8934 against -14.6028 for 2+1+1, lower by 0.291 Ha. But that margin IS the
monotonicity above, at one element: the three candidates order by compactness -- a single 4-shell
at -18.893 (28.8% too deep), 2+2 at -14.893 (1.54% too deep), 2+1+1 at -14.603 (0.44% short) --
so the lower energy is reached by over-binding past the reference, and 2+1+1 is both the least
bound and the closest to it. That matters for consistency as much as for accuracy: the monotonicity above is
precisely the finding that energy is anti-correlated with the physical configuration, so an
energy preference could not have supported one in any case. And the rule is not a constraint we
impose.
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

A demonstration of that kind invites one obvious objection --- that the interface merely stayed
where it was put --- and the revision answers it with a seed ladder. The partition is seeded at a
radius R0 through the core's initial extent, and that seeding fixes nothing thereafter: the
interface is free and where it settles is an output. Seeded at R0 = 1.0 the shell settles at
2.139 a.u.; seeded at R0 = 1.6 it settles at 2.153 a.u. --- a spread of 0.7% on a quantity
nothing in the calculation prescribes, from seeds differing by 60%. The domain charges agree too:
core 2.02 charges at mean radius 0.76 a.u., arriving electron 1.07 at mean radius 4.0 a.u. The
shell radius is an attractor, not a memory of the seed.

We report the rule as established for Li through C and **undetermined from nitrogen onward**,
where a shell holding two charges may be a closed pair or a half-filled tetrahedron and the
model cannot distinguish them.

**The capacities are not fitted either.** The two basic filled packings are established by the
calculations themselves: the antipodal pair, 2, and the dual tetrahedra, 8 = 4+4, the latter
against a single 8-shell that over-binds by 9-22%. Composing rows from those two units alone
gives every row length of the table, not merely the first few. Since a row of length 8k+4 or
8k+6 would need half an octet or three quarters of one and could not be composed at all, the
remainder is constrained to 0 or 2, and the capacities are assembled from the two units directly
as

    cap(n) = 8*floor(n/2)*ceil(n/2) + 2*(n mod 2),

the octet count being the product of the two halves of n and a pair present exactly when n is
odd. This gives 2, 8, 8, 18, 18, 32, 32 and with them every closure - 2, 10, 18, 36, 54, 86, 118
- with the fourth row as 2+8+8 rather than 9+9 and the sixth as four octets. Nothing is imported
and no tile is needed that the packing cannot build.

One point about that expression, which we have changed since drafting this reply. The closed form
is *not* the argument and we no longer present it as though it were: written out, it says only
that the row is filled with as many octets as fit and a pair if one is left over, which is the
composition already stated in the sentence before it. The paper now gives the construction in
words, and where the closed form appears it is labelled a restatement of the shell fill. The
content is the constraint on the remainder --- that a row cannot end in 4 or 6 because neither
half an octet nor three quarters of one is a packing the model can build --- and that constraint
is what closes every row, including the two we cannot otherwise reach. A formula should not be
left to carry weight that the counting behind it is doing.

An earlier draft of this response reported the account as exact through xenon and open at the
sixth row, on the grounds that 32 is not 2+8+8 repeated. That was too cautious and is withdrawn:
32 is four octets, and the expression above reaches every row. Note also what it does *not*
mention - 2n^2. The capacities are built from the pair and the octet and *then* found to coincide
with the hydrogenic count, rather than obtained by decomposing it, so nothing is borrowed from
the theory being compared against.

What we do *not* claim is that the row lengths themselves are derived. The capacities decompose
into pairs and octets uniquely, and that is established; that the third row has length 8 and the
fourth 18 is not. The period lengths are supplied to the configuration generator rather than
produced by it, and that each capacity serves two periods remains the principal open item. The
scope boundary accordingly rests on the two limits we can demonstrate - a build-up that adds
outward cannot reach an inner shell, and the 1s contracts as 1/Z so a fixed grid stops resolving
it.

**One direction on the doubling, offered as no more than that.** Since the reply above admits the
doubling as the principal open item, we should say what the packing account does have to say about
it, because a filling rule has nothing. If each capacity serves k periods, then

    sqrt( cap(n+1) / cap(n) ) = (n+1)/n

exactly when k = 2, the ratios being 2, 3/2, 4/3, 5/4, ... against 4, 2.25, 1.78, 1.56, ... for
k = 1. So k = 2 is the value that converts the quadratic growth of an *areal* capacity into a
*linear* ratio: if a period advances the radius by one unit and the capacity goes as r^2, two
periods are what an areal shell costs, and the doubling is the exponent in r^2 rather than an
extra rule. No other k does this --- k = 3 gives 1.59, 1.31, 1.21, smoother still, so a
preference for smoothness alone would drive k upward without limit, and it is the dimensionality
that picks out two. We put this no higher than a direction: it is a statement about exponents and
becomes physics only when something independent establishes that a period corresponds to a unit
increment of radius. But it is a direction available to a packing account and not to a filling
rule, since the Madelung ordering by (n + l) contains no length for the argument to be about.

**A consequence that can be checked, which we had not drawn before.** If the octet is 4+4, the
approach to closure is not a bare count but a sequence of sites: 1+1, 2+1, 2+2, 3+2, 3+3, 4+3,
4+4. The step before closure is always **4+3** --- one tetrahedron complete, the other one vertex
short. That is fluorine, 2+4+3, and chlorine, 2+4+4+4+3, both as the build-up's own balanced
filling returns them in Table 1. So a halogen is not "one electron short" as a number; it is
short of *one tetrahedral vertex*, a definite vacant site beside a filled tetrahedron, and an
arriving charge closes the octet by occupying it. A filling rule gives p^5 and no site. In the
same vein, a complete outermost octet requires every inner shell already closed at 2 or 8, so
only a noble gas can carry one --- with the caveat, stated in the paper, that the converse fails
for krypton and xenon, whose outermost group is the cap-2 pair.

**Prior art, and one classical result we must concede.** The 4+4 octet is Linnett's double
quartet [Linnett, *J. Am. Chem. Soc.* **83** (1961) 2643], which reads Lewis's cubical octet as
two interpenetrating tetrahedral sets of four; the revision cites it. Linnett separates the two
sets by *spin* --- Fermi correlation within a quartet, Coulomb correlation between them --- and
the point of interest here is that this model has no spin and reaches the same arrangement from
Coulomb repulsion and non-overlap alone. What is new is therefore not the decomposition but a
derivation of it that does not need spin. The concession is the Thomson problem. For four points
on a sphere the Coulomb minimum is the regular tetrahedron (3.6742 against 3.8284 for a square),
so that unit is secure; for eight on *one* sphere it is not the cube but the square antiprism, by
0.33% (19.6753 against 19.7408), and this is now a theorem [arXiv:2609.22077]. We state it in the
paper rather than leave it for a reader to catch. Two things limit its force: the antiprism is
itself a 4+4 --- two squares twisted by 45 degrees --- so what it refutes is the
two-*tetrahedra* reading of a *single* sphere, not the 4+4 decomposition; and this model does not
put eight charges on one sphere, since a single 8-shell over-binds by 9-22% and the octet is
realised over two radii. It does mean the cube must not be offered as a Coulomb minimum, and we
do not offer it as one.

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

**The reviewer is right that the ladder covered one atom, and we have now looked beyond it. What we
found changes our answer, and not in the direction of adding more rungs.** Refining the radial mesh
converges the calculation to the answer of the *spherically reduced* problem, and that reduction --
not the discretisation -- is what limits the accuracy here. The revision measures the cost of the
reduction directly (below): it has the pair count right and the pair geometry wrong, overestimating
intra-shell repulsion by a factor of 1.6-2, and leaving a residual of order a per cent in the
under-binding direction. Set against that, successive sqrt(2) refinements of helium move the total
energy by 0.005 and 0.004 Ha, about 0.2% a step. The model error is an order of magnitude the
larger, so a finer radial grid buys precision in solving the wrong equation. The accuracy-limiting
direction is *angular* resolution, restoring the pair geometry the reduction discards, and we say so
rather than presenting a mesh ladder as though it were the open question.

N = 400 is also what the three-dimensional solver reaches, at 400^3. Holding the radial runs to the
same figure keeps the two halves of the paper comparable, and stops us quoting radial energies at a
precision the three-dimensional model could not be checked against.

**One thing this cost us, which we report rather than leave for the reviewer to find.** Because the
mesh is held fixed rather than converged, a single entry can carry a grid uncertainty exceeding its
quoted error. Argon is the case in point: at the production N = 400 it agrees to +0.1%, the closest
in the table, but on the same domain at N = 724 and N = 1024 -- where the 1s spans 17 and 25 cells
against 10 at N = 400 -- it moves to -511.7 and -511.3 Ha, about 3% *under*-bound, the two finest
rungs agreeing to 0.08%. Both are step-converged: quadrupling the relaxation at fixed mesh moves the
energy by 0.005 Ha. Neon is comparable in size and non-monotone over the same range. Two changes
follow. The manuscript now states that the table is to be read at the level of a few per cent and
that no single entry is claimed to better than a per cent; and we have **removed the claim that the
model over-binds the heavier third-period atoms with the opposite sign of error to Hartree-Fock**,
since the sign of argon's error is a function of the grid. What survives refinement, and what the
reduction predicts, is under-binding.

A word on what
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
- The claim that the model over-binds the heavier third-period atoms with the opposite sign of error
  to Hartree-Fock (R2.8) --- argon's error changes sign between $N=400$ and $N=1024$, so the table
  does not support a sign.

**Added**
- §*Aufbau: how the shell structure forms* — the build-up as the constraint that makes energy
  minimisation select correctly, with the monotonicity measurement (R2.2).
- §*Two numbers, not four: the pair and the octet* — the capacities assembled from the model's
  own packings, the antipodal pair and the $4{+}4$ octet, giving every row length
  $2,8,8,18,18,32,32$ and every closure $2,10,18,36,54,86,118$, without $2n^2$ entering the
  construction at all. The argument is the constraint on the remainder — a row cannot end in 4 or
  6, since neither half an octet nor three quarters of one is a packing the model can build — which
  is what closes every row. The construction is stated in words; the closed form
  $\mathrm{cap}(n)=8\lfloor n/2\rfloor\lceil n/2\rceil+2(n\bmod 2)$ is retained but labelled
  for what it is, a restatement of the shell fill rather than an independent derivation. What is
  claimed is a decomposition of the capacities, not a derivation of the row lengths; the doubling
  of each capacity across two periods is stated as open.
- Within that section, a *direction* on the doubling: $\sqrt{\mathrm{cap}(n{+}1)/\mathrm{cap}(n)}
  =(n{+}1)/n$ exactly when each capacity serves two periods, so $k=2$ is what converts a quadratic
  areal capacity into a linear ratio, and no other $k$ does it. Explicitly labelled a statement
  about exponents and not yet physics.
- The $4{+}3$ result: the step before closure is one tetrahedron complete and the other one vertex
  short, so fluorine is $2{+}4{+}3$ and chlorine $2{+}4{+}4{+}4{+}3$, and a halogen is short of a
  definite tetrahedral *site* rather than of a count. With it, the statement that a complete
  outermost octet requires every inner shell closed — hence only a noble gas carries one — and the
  caveat that the converse fails for Kr and Xe.
- Linnett's double quartet cited as the prior statement of the $4{+}4$ octet, with the difference
  named: Linnett separates the two quartets by spin, and this model has no spin and reaches the
  same arrangement from Coulomb repulsion and non-overlap alone.
- A seed ladder for the three-dimensional build-up, answering the objection that the interface may
  have stayed where it was seeded: $R_0=1.0$ gives a shell at $2.139$~au and $R_0=1.6$ gives
  $2.153$~au, $0.7\%$ apart from seeds $60\%$ apart.
- A paragraph in the Introduction, *The table in two numbers*, putting the pair-and-octet content
  of all seven closed shells in front of the reader before any machinery, with its two limits
  stated there (it fixes how many pairs and octets, not where they sit radially; and it is a
  decomposition, not a derivation).
- Table 1 recomputed with no per-element constant anywhere: three configurations corrected
  (Be $2{+}2\to2{+}1{+}1$, Al and Si to the build-up's own balanced fillings), mean $|$error$|$
  $1.6\%$, maximum $4.5\%$, 13 of 17 atoms within $2.5\%$.
- A statement that the octet's *cube* is a geometric argument and not a result of this paper: the
  spherically reduced solver has no angular coordinate, so its $4{+}4$ is two nested radial
  shells, and the measured shell radii come out uniformly nested rather than paired. With it the
  Thomson concession: for eight points on one sphere the Coulomb minimum is the square antiprism,
  not the cube, by $0.33\%$, and this is a theorem. The antiprism is itself a $4{+}4$, and this
  model does not place eight charges on one sphere, but the cube is no longer offered as a Coulomb
  minimum.
- A provenance note on the anti-correlation table (it runs at 20 000 steps on the default mesh and
  must not be read across against Table 1), both tables having been recomputed after the
  configuration corrections.
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
