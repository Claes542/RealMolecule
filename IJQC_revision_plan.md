# IJQC 1568909 — revision plan

Manuscript: *A 3D Multiphase Continuum Computational Model for Atoms*.
Decision: major revision, due **10 October 2026**.

## Strategy in one line

The science survives; three headline claims do not. Bring the **framing** down to what the
**body text already says**, add the measurements that are missing, and present the packing
result as the finding it is rather than as a weakness to be excused.

Both referees say the method is interesting. R1 calls it "credible that the methods used by the
author are valid approximations to standard QM and therefore could be valuable as efficient
computational techniques", and states the minimum requirement explicitly: *"the author should at
least phrase the ontological claims of RealQM as speculation for now"*. That is an off-ramp, and
it should be taken.

## The key realisation

**Section 5.1 is already honest.** Lines 266–275 say the unconstrained minimum is *not* the
physical configuration, that the 1D solver favours capacity-8, that cap-4 is "selected by best
fit to the total energies", and that a dynamical Aufbau is deferred to later work. R2's
objection is not that this was concealed — he read it — but that the **abstract** still claims
parameter-free prediction of atomic structure. The contradiction is between the body and the
framing, and the framing is what must move.

---

## A. The three claims that must come down

### A1. "In-principle parameter-free" (abstract)

**Contradiction inside the paper:**

- line 165: "Crucially, $r_c$ is thereby *determined by the core solve rather than fitted*, so
  the valence calculation carries no per-element empirical parameter."
- line 384: "Total energies and ionization energies are obtained from their own $r_c$
  calibrations ... **the two error columns reflect two separate fits to one reference each per
  period**."

These cannot both stand. R2 leads with it, and it is the single change that decides the paper.

**Fix.** Keep line 384 (it is the true statement) and rewrite line 165. Say that $r_c$ is
*read off the core density in the two-scale construction*, but that the results reported here
use **two per-period calibrations**, one for total energies and one for ionization energies, and
state what each is calibrated to. Then qualify the abstract: not "parameter-free", but "one
softening radius per period per observable, calibrated to a single reference atom" — which is
still a strong claim and has the advantage of being true.

**Forward statement (do not promise delivery by 10 Oct):** the route to a genuinely
parameter-free $r_c$ is to read it from a *converged all-electron* atom instead of from
reference energies. Mention as the stated path.

### A2. "4s before 3d across the transition series" (abstract) — the worst of the three

R2 asks how the Fig. 1 transition-metal points were obtained and how they establish the claimed
removal order. Checking `atom_simulator.html:452`:

```js
const cfg = capPack(dCount, 4).concat([2]);   // d-shells inner, s^2 outermost
```

The s² shell is placed **outermost by construction**. The run then tests whether the s shell
comes out with the largest radius (`outer_is_s`) and the lowest ionization energy
(`s_least_bound`). In a nested radial model that is very nearly guaranteed by the input.

The code comment is honest — it says this "tests whether this recovers the correct ionisation
order ... that the naive (Madelung-filled) radial d-block cannot". The abstract is not:
**4s-before-3d is an input there, not a result.**

**Fix.** Restate as consistency: *with the d-shell nested inside the s², the model reproduces
the observed removal order, whereas naive Madelung radial filling does not.* Remove "including
4s before 3d" from the abstract's list of results, or mark it explicitly as configuration-dependent.

Two further cautions on those runs: they use 4000 steps (below our own convergence floor — see
the step-ladder note), and the code states total energies there are "floor-carried" and not
trustworthy, only the ordering.

### A3. The ontological claim (R1's "elephant in the room")

R1's objection: the paper defers justification to ref. [1], which is under review, and whose
position is that RealQM is not an approximation to QM but a *correction* to it. He is right that
this cannot rest on atomic results alone.

**Fix.** Take the off-ramp verbatim. Add a short paragraph to the introduction stating that the
present work makes no claim to supersede standard QM; that RealQM is offered here as an
independent continuum model whose accuracy on atoms is an empirical finding; and that the
ontological reading is speculative pending stronger evidence. Do **not** attempt the
DFT-equivalence proof — R1 raises it as a route, not a requirement, and it is not reachable in
three weeks.

---

## B. Straightforward repairs

### B1. Ar: 11.29 eV (Table 3, line 416) vs 8.0 eV (Table 5, line 517)

Same atom, two values, no explanation. It is the two calibrations of A1 surfacing. Once A1 is
fixed, label each table with which calibration it uses and add one sentence cross-referencing
them. Check every other atom appearing in both tables for the same issue.

### B2. p-block ionization energies repeating across periods 2–4 (R2)

Structural, not a table error: with the core absorbed into $r_c$ and $Z_\text{eff}=8$ universal
for every noble-gas valence, each column member presents the **same valence problem**, so the
entries repeat by construction. The model reproduces the *column* and has no representation of
the *period*.

**Fix.** State this explicitly as a limitation with its cause. Also check the Kr→Xe increase R2
flags, which contradicts the claimed down-group trend and may be a genuine error rather than a
structural one.

### B3. Section 5.1 opening sentence

Line 252: "For each atom the shell packing is determined by total energy" contradicts lines
266–275 of the same subsection. Change to *selected by comparison with reference total energies*.

---

## C. Runs required

| # | run | answers | status |
|---|---|---|---|
| C1 | IE grid-convergence ladders (Table 6 covers only He *total energy*) | R2-8 | **not started** |
| C2 | Koopmans $-\varepsilon_i$ vs $\Delta$SCF, explicit comparison | R2-3 | **DONE 2026-09-21 — see below** |
| C3 | Kr→Xe noble-gas trend recheck | R2-5 | not started |
| C4 | Ar cross-table reconciliation | R2-6 | follows from A1 |

### C2 RESULT — the referee is right, and it is worse than a bookkeeping point

`atom_simulator.html?auto=cnod`, bare-nucleus balanced configurations:

| | Koopmans $-\varepsilon_i$ | $\Delta$SCF | NIST |
|---|---:|---:|---:|
| C | 4.85 | **8.74** | 11.26 |
| N | 5.09 | **51.37** | 14.53 |
| O | 6.79 | **12.04** | 13.62 |

**They are not equal.** For C and O the frozen estimate is about half the relaxed one, so Eq. (1)'s
identification of $-\varepsilon_i$ with the optimised neutral−ion difference is unsupported and must
be restated as the frozen-orbital approximation it is.

**And $\Delta$SCF is much closer to experiment** — C 22% low against Koopmans' 57%, O 12% against
50%. The paper reports the worse of the two, and reports it precisely because it needs no ion
calculation, which was the stated selling point.

**But it cannot simply be swapped**, because $\Delta$SCF requires choosing the cation's packing and
for nitrogen neither choice works:

```
neutral   N  2+3+2        E = -54.7368
cation    repacked 2+2+2  E = -52.8492  ->  IE = +51.4 eV  (3.5x too large)
cation    decrem.  2+3+1  E = -55.4088  ->  IE = -18.3 eV  (NEGATIVE - below the neutral)
```

For C and O the two constructions coincide (cation = neutral minus one outer electron) and the
question never arises; for N the balanced rule reshuffles 3+2 into 2+2, a real rearrangement, and
the model cannot say which packing the ion takes. **So $\Delta$SCF inherits the packing-selection
problem** — the same root cause as section D.

**Revision text:** report both columns; state plainly that $-\varepsilon_i$ is a frozen-orbital
estimate and not the ionization energy; note $\Delta$SCF is closer where the cation packing is
unambiguous; and cite the packing-selection problem as why it cannot replace Koopmans outright.
That answers R2-3 with a measurement and ties it to the Aufbau gap the paper already promises to
close, instead of leaving an unsupported equality in the text.

**C2 note.** `__autoCNOd` already computes both, and already records that the cation must be
allowed to **repack**: decrementing the neutral's outer shell gave a spurious state 1.35 Ha
below the true neutral and a *negative* ionization energy. That is a direct, quantitative answer
to R2's question of why $-\varepsilon_i$ should equal the optimised neutral−ion difference — and
the honest answer is that it does not exactly, and the gap is measurable.

**C1 warning.** Do not assume these are a formality. Runs in this project have drifted by
hundreds of kcal/mol long after appearing steady.

---

## D. Turn R2's sharpest objection into the paper's most interesting result

R2: *"the radial model favors cap-8 packing, yet cap-4 configurations are chosen because they
agree better with reference energies."*

Correct — and now **measured**, not merely asserted. Scan of 21 Sep 2026 (`atom_simulator.html?auto=scan`):

| O packing | E (Ha) | vs exact −75.067 |
|---|---|---|
| 2+6 (single shell) | −82.23 | 9.5% over |
| 2+4+2 | −78.33 | 4.3% over |
| **2+3+3 (cap-4)** | **−75.38** | **0.4% over** |
| 2+2+4 | −72.88 | 2.9% under |

and for N (exact −54.589): 2+5 → −58.15, 4+1 → −57.08, **3+2 → −54.74**, 2+3 → −53.25.

**The energy is monotone in the packing.** There is no interior minimum: the model slides to the
most compact arrangement, and the exact answer is *bracketed* between the balanced packing
(0.3–0.4% over) and the sparse one (2.5–2.9% under). So the model cannot select a packing —
not because the selection was done carelessly, but because no variational criterion exists to do
it with.

This upgrades the existing text in two ways: it replaces "cap-8 over-binds by +9% to +22%" with
a two-sided bracket, and it converts "we chose cap-4 by fit" into a structural statement —
**the configuration nature selects is not the one of least energy in this model**, which is
precisely what the deferred Aufbau study would explain. Section 5.1 already promises that study;
this gives it a quantitative motivation.

---

## E. What to say about molecules (R1's questions)

R1 asks whether the method scales to molecules, whether the spherical model is useful beyond
atoms, and whether basis sets could help. There is real material, but it must be stated with its
limits: molecular work in this project reproduces **geometry, forces, and differences within one
composition**, while absolute energetics involving a bare added charge (proton affinity) fail by
several hundred kcal/mol and with the wrong sign. Say that plainly; it is a better answer than
optimism, and it is consistent with the limitations section.

On basis sets: R1's point is well taken and cheap to concede — symmetry-adapted bases could
reduce cost dramatically in periodic systems, and nothing in RealQM forbids them.

On imaginary time: yes, it is a relaxation scheme for the numerics only, with no connection to
path-integral or centroid formalisms. One sentence.

---

## F. Order of work

1. **A1** (r_c) — decides the paper.
2. **A2** (4s/3d) — an abstract claim that is not supported; fix before anything else is drafted
   around it.
3. **D** (packing bracket) — converts the main objection into a result. Cheap; data in hand.
4. **A3** (speculation caveat) — wording, but it is R1's stated minimum for publication.
5. **B1–B3**, then **C1–C4**.
6. Response letter last, written against the finished text.

Not attempted in this round: DFT equivalence; a genuinely parameter-free $r_c$; the dynamical
Aufbau. All three should be named as stated future work rather than promised.

---

## G. PROPOSED NEW STRUCTURE (the "new elements" perspective)

The revision is mostly a **cut** plus two additions. Cutting the ionization-energy material removes
six of Reviewer 2's eight objections (C2/R2-3, R2-5, R2-6, R2-7, R2-8, and half of R2-1) *and*
removes the paper's weakest numbers. What replaces it is measured and parameter-free.

### What the paper becomes

| § | content | status |
|---|---|---|
| 1 | Total energies, 1-2% across the table | **unchanged** — the quantitative validation |
| 2 | Packing: cap-4, and why energy alone cannot select it | **measured** (bracket below) |
| 3 | **Aufbau**: the build-up constraint is what makes selection work | **measured** (Be below) |
| 4 | **Electropositivity / electronegativity** as the valence observable | **measured** |
| 5 | Limitations: light p-block, with the 3D angular solve named as the cure | unchanged |
| — | Ionization energies | **cut or demoted to a short honest subsection** |

### §2 evidence — energy is monotone in packing (measured 2026-09-21)

O: 2+6 -82.23 (9.5% over) | 2+4+2 -78.33 | **2+3+3 -75.38 (0.4% over)** | 2+2+4 -72.88 (2.9% under),
exact -75.067. N likewise. **The exact answer is bracketed and there is no interior minimum**, so the
model cannot select a packing variationally. State this as a structural result, not an apology.

### §3 evidence — and this is the key correction

The anti-correlation is real but CONDITIONAL:

```
Be   4        -18.8926   28.8% too deep   <- core MERGED into the valence
Be   2+2      -14.6074    0.41%           <- physical (1s2 2s2), and the ENERGY'S PICK
Be   2+1+1    -14.6028    0.44%
```

**Energy fails only when it is allowed to dissolve the core.** Among configurations that keep a
proper core it selects correctly (Be picks 2+2 by 0.13 eV). Inheriting the parent's topology --
which is what a build-up does -- never offers the merged option. So Aufbau is not a patch over a
broken variational principle; **it is the constraint that makes the variational principle work.**
That is a far stronger claim than "energy cannot select configurations", and it is testable across
the row (Be+->Be, Be->B, B->C, C->N).

Note this corrects an earlier reading in which Be's physical configuration was taken from the
balanced cap-4 rule (2+1+1) rather than from 1s2 2s2 (2+2).

### §3 evidence — shell radii are OUTPUTS of the free boundary

`atom_simulator` moves each shell boundary M[s] by amplitude continuity (u[M] > u[M+1] -> M++),
the same rule the 3D front implements as cm = u_self - u_other. So the radii are found, not
imposed, and there is **no r_c anywhere** in them:

| el | config | core outer radius (au) | r_core x Z |
|---|---|---:|---:|
| Li | 2+1 | 1.925 | 5.78 |
| Be | 2+1+1 | 1.413 | 5.65 |
| B | 2+2+1 | 0.998 | 4.99 |
| C | 2+2+2 | 0.788 | 4.73 |

Scaling close to 1/Z. **This is the quantity section A1 needs**: a core radius read off a converged
atom rather than calibrated to NIST. It does not make the present results parameter-free -- those
still use the two calibrations -- but it makes the stated forward route concrete and measured
rather than aspirational.

### §4 evidence — electronegativity, parameter-free

Allen configuration energy chi = (n eps_s + m eps_p)/(n+m), computed natively from the shell
energies with no r_c and no anion (unlike Mulliken, which needs EA, and Pauling, which is built
from bond energies):

| | Li | Be | B | C | N | O | F | Ne |
|---|---|---|---|---|---|---|---|---|
| chi model (eV) | 7.26 | 10.18 | 14.49 | 20.68 | 28.81 | 36.69 | 47.93 | 58.39 |
| chi Allen (eV) | 5.39 | 9.32 | 12.13 | 15.05 | 18.13 | 21.36 | 24.80 | 28.31 |
| ratio | 1.35 | **1.09** | **1.19** | 1.37 | 1.59 | 1.72 | 1.93 | 2.06 |

**Ordering is exact across the row**; the gradient is 2.23x too steep; and the error is
**concentrated at the electronegative end**, degrading monotonically toward F/Ne. That is the
documented light-p-block failure (radial 4+4 puts the outer tetrahedron too far out), so the
weakness already has a stated cause and a stated cure. Present the result where it is strong --
the electropositive end, Be 1.09 and B 1.19 -- and state the rest as the known limitation.

**Why chi and not IE.** Ionization is a PROCESS: the ion repacks, and this is a ground-state
model. Measured (C2 above): Koopmans and Delta-SCF differ by ~2x, and for N neither cation
construction works. chi is a **state property of the neutral atom** -- no ion, no relaxation, no
second calibration. It asks the model what it is built to answer.

### Still running / not yet in hand (do NOT write around these)
- 3D shell-radius attractor (seed ladder) and the wrap-around demonstration. These are the
  *mechanism* demonstration for §3; the numbers there carry a ~15% overestimate versus the 1D
  reference (2.21 vs 1.925) because eps = 2h leaves the well 8.8% too shallow at Li's 1s peak.
- C1 (IE convergence ladders) and C3 (Kr->Xe). If the IE material is cut, **both become moot**.

### Risk note
Restructuring three weeks out is safe *here* because the change is dominated by deletion of weak
material, and every addition is already measured. Do not also attempt: the DFT-equivalence proof,
a genuinely parameter-free r_c, or a working dynamical Aufbau. Name all three as future work.

---

## H. THE HALF-SHELL TILING — capacity derived, and the failure boundary predicted

Claes's argument (from the book): a shell divides into two half-shells, each roughly a square;
subdividing a square goes as $n\times n$; hence capacity $2n^2$. The periodic table uses each $n$
twice except the first, $n = 1, 2, 2, 3, 3, 4, 4$, giving **2, 8, 8, 18, 18, 32, 32** and
cumulative 2, 10, 18, 36, 54, 86 -- the noble gases exactly.

| n | per half-shell | capacity | status in the model |
|---|---|---|---|
| 1 | 1x1 = 1 | **2** | the antipodal core pair |
| 2 | 2x2 = 4 | **8 = 4+4** | **measured**: the closed octet is 4+4, a single 8-shell over-binds 9-22% |
| 3 | 3x3 = 9 | 18 = 9+9 | **not expressible** |
| 4 | 4x4 = 16 | 32 | not expressible |

### What this buys, §packing
The 4+4 octet stops being an empirical finding and becomes the $n=2$ case of a counting argument,
and the cap-2 core becomes the $n=1$ case. Both were previously stated as measured results
(Section~5.1). Deriving them removes exactly the kind of "chosen because it fits" that R2 objects to
-- and it costs no parameter, since $1,4,9,16$ are counts, not calibrated numbers.

### What this buys, §limitations -- the stronger point
**The subdivisions are independent: 9 cannot be assembled from 4s.** A 3x3 tiling is not built from
2x2 blocks. The solver's vocabulary is exactly `1x1` and `2x2` (`atom_simulator`'s own note: cap-4
shells, "2 + 21*4 = 86"), so its period-4 configurations such as 2+4+4+4+4+... sum to the right
electron count while being **the wrong structure** -- they are not 9+9.

Consequences, in order of value:
1. **The range of validity is principled, not empirical.** The method covers exactly what the
   $1\times1$ and $2\times2$ tilings span: **H through Ar**. Everything the paper does well lies
   inside that.
2. **The failure boundary is predicted.** $n=3$ is first required at $Z=19$; the documented
   breakdown (Section~\ref{sec:dblock}) sets in at $Z \approx 20$--21.
3. **The CHARACTER of the failure is predicted too.** A missing tile gives a structural break, not
   accumulating error -- and the observed behaviour is that energies *explode positive* rather than
   degrading through the 20s. Quantitative error accumulates; a wrong structure breaks. The present
   text attributes the wall to interpenetration (3d more contracted than 4s), which is a
   description of the symptom; the tiling argument predicts both the location and the abruptness.
4. **It makes the heavy-element tables visibly an extrapolation** past the point where the
   structure is known to be wrong. R2 pressed on exactly those tables (repeating p-block IEs,
   Kr->Xe). Saying so directly is better than defending them.

### Honest limits of the argument
- It explains capacities and the repetition ($n$ used twice), but the *reason* each $n$ serves two
  periods is not derived here -- in standard QM that is the $n+l$ rule. State as open.
- It is **not testable in the 1D solver**. A 9-per-half-shell is an angular arrangement and the
  radial model has no angular structure: it would report 9-in-a-shell as over-bound exactly as it
  reports 8-in-a-shell as over-bound, while the paper argues the physical octet *is* 4+4 at one
  radius angularly separated. Same wall as the light-p-block ionization ceiling, same cure (3D
  valence solve).
- A measured size/radius version was tried (s = half the shell thickness, R = mean radius, from the
  free boundaries) and **does not work**: shells holding 4 electrons give s/R = 0.36--0.63 where
  tetrahedral packing needs 0.816, and there is no plateau structure. The innermost shell's
  s/R = 1 is definitional (R_in = 0), not a confirmation. Do not use that version.

### Cheap check worth doing
Run Z = 17--22 and show the energies turn sharply at the period-4 boundary rather than drifting.
The argument predicts a break at $Z = 19$; the recorded onset is $Z \approx 20$--21.
