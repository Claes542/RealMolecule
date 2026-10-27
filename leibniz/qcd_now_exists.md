## Three Proofs in One Announcement

On 26 October OpenAI announced that its internal models had produced three research results. In mathematics, a proof of the local smoothing conjecture in every dimension, carrying the Kakeya conjecture and Stein's restriction conjecture with it. In computer science, a claimed disproof of the Exponential Time Hypothesis, formalised in Lean. And in physics, a rigorous existence proof for certain four-dimensional quantum field theories — including quantum chromodynamics and a completed Standard Model.

Take the first two seriously. Kakeya has stood since 1917 and has resisted everyone. A disproof of the Exponential Time Hypothesis, if it survives, rearranges complexity theory. These are truths of the kind Leibniz said a machine could produce: consequences drawn from what has already been granted, and now drawn at a level that was not expected this soon.

It is the third one I want to sit with, because of what it does and does not say.

## Existence Is Not Computability

A rigorous existence proof for QCD establishes that the theory is a well-defined mathematical object. That is a real achievement and a long-standing open problem. It does not produce the binding energy of a deuteron.

Quantum chromodynamics was written down in 1973. Fifty-three years later there is no lattice calculation that delivers nuclear binding energies at physical quark masses without ingredients fitted to the answer, and the effective field theories used instead carry parameters calibrated on the nuclei they are meant to explain. This is not a controversial reading; it is the ordinary state of the subject, familiar to anyone who works in it.

So after 26 October the position is this. The theory of the strong interaction is — if the proof holds — **rigorously proved to exist**, and it still cannot tell you why helium-4 binds by 28.3 MeV. Those two facts are entirely compatible, and holding them together is the whole point.

## The Same Shape, Again

Readers of this blog will recognise the structure, because it is the one that turns up everywhere once you look for it.

Schrödinger's equation is believed exact for atoms and molecules, and is unsolvable for almost all of them: the wave function of forty-two electrons lives on a 126-dimensional space. What is used instead is density functional theory, which scales — and which requires its user to name an exchange–correlation functional, chosen from several hundred, most of them fitted to measured data. The Navier–Stokes equations are believed exact for ordinary fluids, and resolving turbulence at the Reynolds numbers of an aircraft is beyond any computer that will be built. What is used instead is a turbulence model with constants calibrated against measured flows.

In each case the exact theory is out of computational reach and the practical substitute carries an ingredient supplied by hand. And in each case the consequence is the same: **what gets tested is the ingredient, not the theory.** When a number disagrees there is a functional to change or a constant to retune; when it agrees, agreement was available by construction.

QCD now joins that list in its purest form. Not "we think it is right but cannot check" — rather "it is *proved* to exist, and still cannot be made to produce the number." Proof of existence removes the last doubt about the object while changing nothing about its silence.

## What Leibniz Would Have Said

He drew the line himself, three hundred years ago, and the announcement illustrates it better than any argument could.

There are truths of reasoning, whose opposite is impossible, which follow from what has already been granted. And there are truths of fact, whose opposite is possible, which must be checked against how the world is. A machine that manipulates symbols correctly can produce every truth of the first kind and none of the second.

Three magnificent truths of reasoning were produced last week. Not one of them weighs a nucleus. The existence proof grants QCD its mathematical standing; whether QCD accounts for the matter in your hand is a different kind of question and no quantity of proving reaches it. The machine did exactly what Leibniz said it could, including the part about what it could not.

## Two Cautions, Because They Matter Here

These results are days old and extraordinary, and extraordinary claims in mathematics are settled by the community, not by the announcement. That process has started — there is already a human audit of the proofs circulating, and a paper on what happens to mathematical knowledge when proof-checking becomes free and adjudication becomes the scarce thing.

And formal verification deserves to be understood precisely. Lean establishes that a proof follows from the statements as formalised. It does not establish that the formalisation says what the informal theorem meant. That translation is still human work, it is where the subtle errors live, and it is the reason "verified in Lean" is a strong claim about one thing and silent about another. It is the same distinction as above, one level up: the checker handles the truths of reasoning, and somebody still has to look at the formalisation and judge whether it corresponds.

## What Actually Changes

If deduction is becoming cheap — and three results in one announcement suggests it is — then the scarce good is no longer derivation. It is confrontation with fact.

Which gives a criterion for where effort is worth spending, and it is not the one the field currently uses. A theory that cannot be computed outright, and whose practical versions carry ingredients tuned to the answer, gains very little from cheap deduction: it will acquire better proofs of its own consistency and better fits to the data, and remain exactly as untested as before. A theory that can be computed, with nothing left to select, gains enormously — because the one thing now cheap is producing consequences that could turn out wrong.

So the lesson of the QCD proof is not about QCD. It is that proving a theory exists and showing a theory is right are different activities, that machines have just become spectacularly good at the first, and that the second has not moved at all. The bottleneck has shifted, and it has shifted towards the part Leibniz said was ours.
