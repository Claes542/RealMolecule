Here is a question a schoolchild can ask and nobody can answer in one line. The rows of the periodic table have lengths

**2, 8, 8, 18, 18, 32, 32, 50, …**

Why those numbers? Why does each one except the first appear twice? And why is the eighth row's 50 a prediction rather than an observation?

## The textbook answer, and exactly where it stops

Orbitals have capacities *2(2ℓ+1)* — so 2, 6, 10, 14 for s, p, d, f — and they are filled in the **Madelung order**, by increasing *n+ℓ* and, within a tie, by increasing *n*. Add the capacities up in that order and the row lengths come out right.

Two halves of that have very different standing, and separating them matters, because the standard account is stronger than its critics sometimes allow. The capacities **are derived**. A central field is rotationally symmetric, the ℓ-th spherical-harmonic representation has dimension 2ℓ+1, and spin-half with the Pauli principle doubles it: 2(2ℓ+1) is a theorem, not a pattern. So 2, 6, 10, 14 are genuinely obtained.

The *order* is the part that is not. The Madelung rule is no theorem of the Schrödinger equation; Löwdin said so in print in 1969 and named it an open problem, and it has stayed open. Demkov and Ostrovsky did derive it exactly, but for a specially chosen potential interpolating between Coulomb and short range — which shows the rule is consistent with some Hamiltonian rather than following from the atomic one. Numerical self-consistent field calculation is the strongest practical answer: solve the atoms and the observed configurations largely come out, with about twenty anomalies — chromium, copper, niobium through silver, lanthanum, gadolinium, platinum, gold — which fall exactly where the Madelung order fails. And the qualitative reason for the ordering is sound: low-ℓ orbitals penetrate the core, so screening binds them more tightly and lifts the hydrogenic ℓ-degeneracy. That is derived in direction. It is not derived in the quantitative crossings that make the sequence 2, 8, 8, 18, 18.

So the honest summary is that **standard quantum mechanics derives the capacities and not the sequence**, and the doubling inherits the standing of the weaker half. That is narrower than the flat claim that the table has never been derived from quantum mechanics, and better for being narrower.

Which is not nothing. But it is worth knowing which it is.

## A different answer: two packings and no third

Take the question as geometric instead. Put unit charge clouds on a shell and let them find their own places under mutual repulsion. Two filled arrangements appear, and there is no third: the **antipodal pair**, 2, and the **tetrahedron**, 4.

And now the one step that does the real work. A single tetrahedron has no inversion centre. It is *directional* — those four vertices are carbon's four bond directions, which is why carbon bonds at all. A closed shell has to be spherically symmetric, so a lone tetrahedron cannot close one. But a tetrahedron has a *dual*: the same figure inverted through the centre, a distinct arrangement. Their union is the eight corners of a **cube**, and the cube *is* centrosymmetric.

**So 4 cannot close and 8 can.** And the sequence stops there, because the cube admits no further dual — it is already its own inversion. The pair is centrosymmetric from the outset, which is why 2 closes without doubling at all.

Two filled packings, 2 and 8 = 4+4, and a reason why there is no third.

## The rows, in that unit

Compose the rows from those two and nothing else:

```
row 1    2                 length  2    closes at   2    He
row 2    8                 length  8    closes at  10    Ne
row 3    8                 length  8    closes at  18    Ar
row 4    2 + 8 + 8         length 18    closes at  36    Kr
row 5    2 + 8 + 8         length 18    closes at  54    Xe
row 6    8 + 8 + 8 + 8     length 32    closes at  86    Rn
row 7    8 + 8 + 8 + 8     length 32    closes at 118    Og
```

Seventeen shells instead of thirty-one, and the noble gases fall where they fall. The eighth row's 50 is composable in the same unit — 2 + 6×8 — which is why the scheme permits it; the table itself stops at 118, so that row remains an extrapolation either way.

## The sharp part, and the honest part

**The sharp part.** Three of the four orbital capacities never occur in this account. There is no 6 and no shell of six; no 10; no 14. Not as approximations, not as limits, not under other names — the packing simply has no such arrangement, and a row of length 8k+4 or 8k+6, which a sub-shell of six or ten would require, cannot be composed at all. What survives is 2, and even that survives as a different object: not the *ℓ=0* harmonic but two unit charges at opposite poles. Geometry rather than angular momentum. The two accounts agree on the row lengths and on nothing underneath them.

**The honest part.** The word *doubling* is doing two jobs here and they must be kept apart. The *octet* doubling — a tetrahedron together with its dual, inside one shell, giving the cube — is explained, by the inversion argument above. The *period* doubling — a whole capacity serving two consecutive rows, 8 then 8, 18 then 18, 32 then 32 — is **not explained**, not by inversion and not by anything else offered here. It is the open item. The two concern different objects, one within a shell and one across rows, and explaining the first is not a down-payment on the second.

## So what is the best answer?

That depends on what you want from it.

If you want the numbers, the orbital account delivers them and has for a century, and it derives its capacities properly. If you want to know why the *sequence* runs 2, 8, 8, 18, 18, it hands you the filling order, and there the derivation runs out.

The packing account also derives capacities, and different ones, from geometry with nothing fitted: why 2, why 8, why not 4, and why the list of filled packings terminates. It does not answer why each capacity serves two rows either. **So neither account derives the doubling, and the two differ in which capacities they derive** — which is the useful part, because that difference is testable. One says shells of 6, 10 and 14 exist; the other says no such arrangement is available at all. They cannot both be right about what lies under the row lengths, and the row lengths alone cannot tell them apart.
