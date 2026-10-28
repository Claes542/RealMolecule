The Navier–Stokes problem has been advertised for more than twenty-five years as one of the great open questions of mathematics, with a million dollars attached. When proposed resolutions appear, substantive public mathematical discussion is harder to find than one would expect. This post is about why, and about what the problem is asking.

## What is claimed, and what is not

The concession goes first, because it removes the only cheap answer.

The Hoffman–Johnson constructive computational theory of turbulent Navier–Stokes flow does not prove global smoothness and does not claim to. What it delivers is constructive existence and qualitative uniqueness: computed weak solutions for specific data, with the error in each mean-value output bounded by duality against a computed solution of the adjoint equations, and with mean drag and lift converging to a few per cent on meshes on which the velocity field does not converge and never will. The solutions have gradients growing like *ν^(−1/2)* while their velocities stay bounded. That is not a discharge of Fefferman's statement (A), and no prize is being claimed.

So the question is not why the prize has not been awarded. It is: **if this does not reach the heart of the problem, what is the precise mathematical objection?**

## The heart, in the formulation's own words

I do not have to argue about whose question matters, because the official statement says it. The four alternatives are offered, in Fefferman's words, *to give reasonable leeway to solvers while retaining the heart of the problem* — and the Institute's own description of the problem locates that heart in turbulence, in the fact that the equations are believed to contain the phenomenon and that nobody can say what they contain.

The claim is then easy to state: the leeway was used to reach turbulence. The material is not new. It was published in 2007, in a chapter of a book written with Johan Hoffman, and has been publicly available ever since.

To my knowledge no published mathematical critique has explained why this fails to reach that heart. Direct approaches to leading specialists have not produced a substantive mathematical response — though in one case I asked for none, and that should not be counted as silence.

## The explanation that needs no speculation

Several explanations suggest themselves, and most are guesses about other people's minds: specialisation, institutional habit, the sheer effort of assessing an unconventional argument with nothing established to reward the assessment. I will not press any of them, because one explanation is not a guess at all. It is a mechanism, and it is visible in the formulation itself.

**The prize is awarded for a statement of a particular logical form.** It asks whether smooth data give a solution that stays smooth, or whether the velocity blows up. A result of a different form cannot win it however good it is, and — more to the point — has no established place in the conversation the prize organises. Criteria sharp enough to adjudicate are, for that very reason, not built to ask whether a different formulation better captures the underlying question.

That is nobody's fault. It is what happens when a scientific problem is given a legal form so that it can be judged.

## And the bar for a refutation is low

This is worth saying plainly, because it separates the case from the usual one. The results are not merely published; they are **executable**. The meshes, the outputs, the adjoint solutions and the duality-weighted error bounds are reproducible by anyone with a machine. Refuting them does not require reading a long argument sympathetically. It requires pointing at the step where a bound fails, or running the computation and getting something else.

## The same standard, applied to the new result

OpenAI has presented finite-time blow-up constructions for Euler and for Navier–Stokes, formalised in Lean. I take the proof to be correct, and nothing above depends on whether it is. It settles a question open for decades and it is a real result about the model.

An analysis of its mechanism exists, and I wrote it, so what follows is not a complaint about an absence but a request that something present be engaged with. What is asserted is that the *gradient* and the vorticity time-integral diverge, with nothing said about the velocity becoming unbounded. That is exactly the signature of the computed turbulent solution, whose gradients diverge while its velocity stays bounded — so the two accounts agree on the fact. But **the same quantity diverges for opposite reasons**. In the turbulent flow the gradient grows because the cascade carries energy to smaller and smaller scales and the energy *leaves*; that divergence is the mechanism of dissipation, which is the drag. In the construction the gradient grows because the energy *cannot* leave — the solution is classical and conserves it exactly — so a growing gradient can be accommodated only by concentrating what is already there. One divergence is energy departing, the other is energy piling up. The symptom is shared and the physiology is opposite.

They differ in kind of event as well. In the turbulent solution the unbounded gradient is a *persistent condition*: the solution goes on existing, as a weak solution, dissipating, for as long as one cares to run it. In the construction it is a *terminal event*, after which the theorem says nothing. And there is no viscosity anywhere in the construction, hence no boundary layer — which is where drag and lift are produced.

If that reading is wrong, the correction would be worth more to me than the prize.

## What would be valuable

Not a verdict. Two things.

If the constructive theory does not reach the heart of the problem, the precise mathematical objection: a step, a bound, a claim that does not hold. The absence of a published critique does not establish correctness, just as the absence of public discussion does not prove that no assessment has occurred.

And if the blow-up construction is a fundamental advance, an account of what it tells us about fluids — its mechanism, its dependence on viscosity, its relation to instability and to turbulence.

Mathematics advances by producing proofs. It also advances by examining what its problems mean and what their solutions tell us. **A Millennium Problem that is only ever answered yes or no risks becoming a monument to its own formulation.**
