## Three Existence Proofs in One Month

This October has produced an unusual run of existence results. A finite-time singularity for the Navier–Stokes equations under a particular force. A rigorous construction of certain four-dimensional quantum field theories, including quantum chromodynamics. And, on a far smaller scale, I was reminded this week that a minimiser for the charge-density model I work on exists as soon as one keeps the interface term the theory already carries — compactness in the space of functions of bounded variation, lower semicontinuity, done in a line.

All three are real theorems. Not one of them tells you anything about matter.

The singularity result says nothing about the drag on a wing. The construction of QCD says nothing about why helium-4 binds by 28.3 MeV. And my minimiser says nothing about where the interface sits, what shape the cell takes, or what the energy is. In each case the information content about the physical world is zero, and that is not a criticism of the proofs. It is what existence proofs are.

## Why Pure Existence Is Cheap

An existence theorem is a statement about a space of possibilities, not about a configuration. The direct method of the calculus of variations wants three things: an energy bounded below, enough coercivity that a minimising sequence cannot escape, and lower semicontinuity so that the limit is no worse than the sequence. Supply those and a minimiser exists. You learn that the set of candidates is not empty. You learn nothing whatever about which candidate it is.

And it gets cheaper the moment you are willing to regularise. Add a perimeter penalty and the admissible configurations have bounded boundary measure, so compactness is immediate. Or discretise: a finite grid with a finite set of labellings is a compact finite-dimensional problem, and existence follows by inspection. Both are legitimate, both are standard, and both leave you exactly where you started if what you wanted was a number.

## What the Clay Problem Asks, and Why That Is the Wrong Question

The Clay Navier–Stokes problem asks for existence and smoothness of solutions for all time. It is a beautiful question and it is the wrong question for anybody who wants to know how a fluid behaves. A proof either way — smooth forever, or a singularity in finite time — leaves the practising aerodynamicist precisely where they were. It was posed in an idiom in which existence was the frontier, and in that idiom it made sense. The frontier has moved.

What one actually wants from a turbulent flow is a drag, a lift, a mean pressure. And what one wants to know about those is not that they exist but what they are, and how far from the truth the computed value lies.

## One Deleted Rule

The objection is a hundred and forty years old, and it is worth being precise about whose it was. Kronecker — “God made the integers, all else is the work of man” — is usually remembered as having objected to infinity. What he actually refused was an existence claim that declines to produce its object. That turns out to be exactly the right target. Under the Gödel–Gentzen translation, which embeds classical arithmetic inside constructive arithmetic, the universal quantifier passes through untouched, and so do conjunction and implication. The two connectives that have to be doctored are “there exists” and “or”. Kronecker had put his finger on the one place where classical and constructive mathematics genuinely part company.

The parting is a single inference rule. Constructive logic is classical logic with ¬¬A → A deleted, and nothing else changed. One rule. Everything follows from that deletion: that a proof of “A or B” must say which, that a proof of “there exists an x” must hand over the x, and therefore that the entire genre of proof-by-contradiction-of-nonexistence stops counting. It is not a rival mathematics; it is classical mathematics minus an assumption, so every constructive theorem is also a classical one. Fewer theorems, each saying more. **To exist is to be exhibited.**

Set beside that, the quantifier structure of the Clay problem is worth looking at. It asks: for all smooth divergence-free initial data of finite energy, and for all later time, the solution stays smooth. The outer quantifier ranges over a function space, which places the statement above the level at which any sound and complete proof system exists at all. “A minimiser exists” is of the same kind — a second-order existence claim, obtained by compactness and by choosing a convergent subsequence. Whereas “this computed output lies within δ of the true one” is a finite, decidable statement about an object one has in hand. The distance between those two is the whole height of the logical hierarchy. Replacing the first by the second is not a weakening of the question. It is a change of logical type.

There is also a sharper way to say why pure existence says nothing, and it is technical rather than rhetorical. Constructive existence of a real number is not the demand that something terminate in finitely many steps, because a real number never does. It is the demand for a modulus: for every tolerance you name, I tell you how many steps suffice, and I can compute that number. A Cauchy sequence without a modulus is not a constructive real at all. Specker settled this in 1949 with a computable, bounded, increasing sequence of rationals whose limit is not computable — so monotone and bounded gives convergence only in the classical sense, the limit existing while no rate exists. That is exactly what compactness hands you: a convergent subsequence with no rate of convergence. **The missing modulus is the missing information**, and it is the same missing thing in the calculus of variations as in arithmetic.

This is also where the standard I am asking for goes past the logicians’. Brouwer wanted a witness. What computation makes it reasonable to want is a witness together with a certified bound on the quantity one actually cares about — a modulus for the output rather than for the solution, which is why it can be had in cases where the stronger thing cannot: the bound is computed afterwards rather than proved in advance. Logic has a mature branch running the converse operation, proof mining, which extracts explicit rates from classical proofs that never mentioned one. The direction of travel is the same in both. Existence was never in dispute. The modulus is the content.

## Constructive Existence

There is a different standard available, and it is not new: **compute the solution, and bound the output by a quantity the computation itself produces.** Not “a solution exists” but “here is one, and here is how far it is from the truth.”

Concretely, for each output of interest one linearises a dual problem about the computed solution, solves it, and evaluates its stability factor; the error in the output is then bounded by the residual weighted by the dual, and the mesh is refined where that product is largest. Two things follow that a pure proof cannot give. The error is *bounded* rather than guessed at by comparing successive meshes and hoping. And adaptivity and error control turn out to be one mechanism rather than two — the quantity that tells you how wrong you are also tells you where to work.

There is a precedent for accepting this as proof, and it is a good one. The four-colour theorem was settled by computation, and the settlement was accepted because the computation was reproducible and its error controlled. That is the whole of the standard.

And it brings a reformulation with it. A problem can be *well posed in the quantities that are observable while remaining ill posed in the field that carries them*. Mean drag and mean lift for turbulent flow past a bluff body come out to a few per cent on meshes of a few hundred thousand points — small enough for a personal computer, and orders of magnitude coarser than resolving the Kolmogorov scale would require. On those same meshes the pointwise velocity field is not converged and never becomes so. Both facts hold at once. Turbulence stops being the thing that makes fluid mechanics incomputable and becomes the thing that makes the pointwise field irrelevant.

## And It Supplies an Object

There is a third thing constructive existence gives, beyond a number and a bound, and it is the one that has actually produced understanding.

A computed solution is an *object*: a particular flow or a particular charge configuration, held in full, every field at every point, open to being plotted, sectioned, differenced and stared at. An existence proof supplies nothing of the kind. You cannot inspect the QCD vacuum to see why nuclei bind; the construction establishes that the theory is there and leaves you with no configuration to look at.

And inspection is not a figure of speech. The resolution of d'Alembert's paradox and the account of how a wing generates lift were both obtained by looking at computed turbulent solutions — seeing where the instability acted, what it generated, and which part of the surface carried the drag. Both are stated symbolically afterwards, as arguments about stability and about vorticity, not as tables of numbers. The computation did not replace the analysis. It supplied the object the analysis was about.

The same thing happened to me this week in the quantum case, and the parallel is almost verbatim. Asking *which part of the cell carries the kinetic energy* — the analogue of which part of the surface carries the drag — turned a puzzle into a mechanism. The computed valence density turned out to be flat to within eight per cent across the whole outer region, with its peak not at the ion but interstitially, between ions. A density that flat has almost no gradient, so a kinetic term that charges for variation collects almost nothing from it: the whole kinetic energy is one and a half per cent of the total. And more than half of even that little comes from a region occupying under nine per cent of the volume — the climb out of the depletion at the ion core, which is the only real gradient in the cell.

None of that was derived. It was seen, by plotting the thing and looking, and only then written down. An existence theorem for the same model — which, as noted, takes a line — would have yielded none of it.

## The Same Standard in Quantum Mechanics

None of this is special to fluids, and the quantum case makes the point more sharply because the contrast is starker.

For a ground state, pure existence again comes by the direct method and again says nothing. The constructive version is available too, and the mathematics is written down: for nonlinear eigenvalue problems of exactly the kind electronic structure produces, one has a free upper bound from any conforming calculation, a post-processing step that costs one linear solve on a finer space, and an error estimate from the difference. I tried it this week on hydrogen, where the answer is known, and the estimate captured 98 per cent of the true error using nothing exact — while improving the energy by a factor of fifty.

So a properly evaluated calculation does not report a number. It carries a theorem: the energy of the computed state bounds the true minimum from above, and the distance is estimated by a quantity the computation produced.

Against that, consider what computational practice usually reports. Convergence tolerances. *The iteration stopped moving to twelve decimal places.* That is a statement about the solver, not about the distance to the truth, and the two are unrelated: I had a calculation this week that satisfied its constraint to four parts in a thousand billion while being sixty-four per cent wrong. Every tolerance was met. Nothing was true.

## What Changes When Deduction Becomes Cheap

If machines can now produce existence proofs at the rate of three in a month, then existence proofs are no longer the scarce good. A proof that an object exists will become something one orders rather than something one celebrates.

What stays scarce is a number with a bound on it. That is not a deduction; it is a construction plus a confrontation with the possibility of being wrong, and no amount of symbolic facility supplies it. Which gives a criterion for where effort is worth spending, and it is not the criterion the fields currently use. A theory that cannot be computed, and whose usable versions carry ingredients tuned to the answer, gains very little from cheap deduction: it will acquire better proofs of its own consistency and better fits to the data, and remain as untested as before. A theory that can be computed, with a bound attached, gains everything — because the one thing now cheap is producing consequences that could turn out wrong.

Pure existence says nothing. Constructive existence says everything. That was true before this month and it will be true after, but it has rarely been so easy to see.
