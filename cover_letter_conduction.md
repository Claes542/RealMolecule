Claes Johnson
claesjohnson@gmail.com

To the Editors, *Journal of Physics: Condensed Matter*

Dear Editors,

I submit for consideration **"RealQM on Conduction"**, an original research article of 26 pages
with one figure and an appendix of computational details.

**The result.** The article computes what it costs to advance the electron pattern of a solid by
one lattice site -- the activation energy for charge transport -- directly from an electronic
energy, for four arrangements: a chain of sixteen hydrogen atoms, 394 meV per electron; a cable
of sixteen lithium ions with the carriers standing off closed cores, 246 meV; that same cable
with its carriers stiffened, 20 meV, which is below room-temperature kT so the charge is no
longer held; and a simple cubic lattice of seventy-two hydrogen atoms. From the same landscapes
follow threshold fields of 5.2, 3.2 and 0.26 GV/m, the field at which the barrier disappears
being a property of E(d) alone. Closing the prefactor with an attempt frequency built from the
same landscape puts the first two arrangements at 0.1 and 10 S/m, bracketing intrinsic
germanium.

**Nothing in this is fitted.** No relaxation time, transfer integral, effective mass or
pseudopotential enters, and no quantity is adjusted to reproduce a measured one. Three
quantities then land in measured ranges without having been aimed there: the barriers against
semiconductor activation energies, the attempt frequencies at lattice-vibration energies, and
the threshold fields against dielectric strengths. The chain's landscape is also not the assumed
one -- scanned across a period, its minimum lies near a quarter spacing rather than at registry,
so advancing one site is a two-step passage over unequal barriers.

**The basis is not the textbook one, and I state that at the outset.** RealQM represents an
N-electron system by N real-valued wave functions on non-overlapping supports in
three-dimensional space, with non-overlap in place of antisymmetry, and carries the physics in
one functional: Coulomb potential energy without self-repulsion, plus kinetic energy. The reason
this computation appears not to have been done before is structural: the textbook N-electron
state is a function on a 3N-dimensional configuration space, so every practical route first
reduces the dimension and then repairs the reduction with modelled parameters -- and the
transport energy comes out of those parameters rather than out of the two terms it is made of.
Working in three dimensions from the start removes the need for them. I recognise that referees
will wish to weigh the framework as well as the numbers, and the article is written to make both
inspectable.

**What it does not claim.** The conductivities are lower bounds, the rigid slide being one path
between registries. The attempt frequencies are overestimates, built on a curvature the scan
shows to be too sharp. The lattice barrier is an estimate, the windowing that removes end
effects in the other cases not being applied to it. No resistive conductivity is obtained: a
minimisation over real densities has neither a current as an object nor irreversible dynamics.
And each geometry is specified rather than found, so what is computed is a constrained minimum.

**Reproducibility.** The solver runs in a web browser with nothing to install or compile. Every
configuration in the article is a query string, so each calculation is reproduced by following
one link; the links are in the appendix and the code is public. Every energy difference is taken
between two configurations of one system at one mesh and one nuclear cutoff, so that the large
common errors cancel, and the controls on that cancellation are reported in full.

The manuscript is original, is not under consideration elsewhere, and I have no competing
interests to declare.

Yours sincerely,

Claes Johnson
