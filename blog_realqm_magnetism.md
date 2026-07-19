# Magnetism in RealQM: How Far Can Charge in Real Space Take You?

**Claim in one line:** magnetism — the moment of an atom, its response to a field, even the electron's *g = 2* and the two spots of Stern–Gerlach — comes out of charge densities moving in ordinary three-dimensional space, with no relativity; and the one place it *stops* is exactly where physics says it should.

## The starting point, and the problem

RealQM reformulates quantum mechanics as charge densities in real 3D space: each electron is a cloud of charge on its own territory, and the ground state simply minimizes the ordinary Coulomb energy. It reproduces the periodic table, chemical bonding, reactions, condensed phases — all from that one idea.

But there is a catch built in. RealQM's ground states are *real-valued*, and a real charge density carries **no current**: nothing is moving. And magnetism *is* charge in motion. So in its base form RealQM has no magnetism at all. The honest question is: can you get it, and how far?

This post follows that question to the end — including the wall it hits.

## Charge going in circles is a magnet

The fix is minimal and natural. Let the charge cloud carry a **phase that winds in space** — charge literally circulating, going in circles rather than sitting still. That circulation is a real electric current, and a current loop is a magnet. Out comes a magnetic moment, quantized by how many times the phase wraps around.

Two things make this more than a story. First, a small solver actually runs it: a circulating electron cloud holds its moment stably, conserving everything it should. Second, switch on a magnetic field (the ordinary way, through the vector potential) and the circulating cloud **reacts correctly** — its energy splits by exactly the Zeeman amount, to four decimal places, while a *non*-circulating cloud sits inert. So a charge density in real space feels a magnetic field and responds as a moment should. This is ordinary magnetism, rebuilt from charge in motion, no spin and no relativity invoked.

## The electron inside the nucleus carries no moment — and that's a feature

In the RealNucleus picture a nucleus is protons and electrons bound by the electric force. The classic objection that killed that idea in 1932 was magnetic: an electron squeezed inside a nucleus should carry a huge magnetic moment — about a thousand times what nuclei actually have.

In a charge-density theory the answer falls out. The moment is the *current's*, and RealQM computes the confined electron as a **flat, motionless** cloud — no circulation, hence **no current, hence no moment**. Nuclear moments then come out at the small scale actually observed. The thousandfold overshoot never happens, because there is no built-in "intrinsic" moment to carry — only the current, and a flat electron's current is zero. Strikingly, it's the *same* flatness that made the electron's mass irrelevant to nuclear binding: one property answers two of the old objections at once.

## Spin, and *g = 2*, without relativity

The hardest case is spin — the two-valued moment behind Stern–Gerlach's famous *two spots*, and the electron's *g = 2*. Textbooks get *g = 2* from the relativistic Dirac equation, so you might think relativity is unavoidable.

It isn't. Give the charge cloud a two-component (spinor) structure and write its motion in the natural first-order way, and *g = 2* **emerges** — it is a fact about how spin-½ objects rotate (the geometry of the rotation group), not about relativity. An electron with no orbital motion at all then splits, in a field, into **exactly two levels with no middle** — Stern–Gerlach — entirely non-relativistically. This is a genuine result: the thing that looks most like "esoteric quantum magic" turns out to be geometry.

## Where it stops — stated plainly

Here is the wall, and reporting it is part of the point. The single-*atom* moment works. But a **magnet** — a piece of iron, a closed electron shell — is *collective*: many atomic moments locking together. That locking is the **exchange interaction**, and RealQM's geometry does not supply it.

We tested the simplest case: two electrons in a closed shell should pair to *zero* net moment (they should repel a field, not follow it). In RealQM they don't — left alone they align *with* the field, the wrong way. And trying to force them to pair through the shared boundary between their territories actually costs energy, so geometry pushes them the wrong way. The clean statement this earns: RealQM's picture reproduces the **spatial** side of the exclusion principle (why the periodic table looks as it does) but **not its spin side** (pairing, exchange, permanent magnets). Single-particle magnetism: yes. Collective magnetism: not without something more.

## What it means

So magnetism, read through RealQM, splits cleanly. The **moment of a single atom** — its circulation, its response to a field, its spin, even *g = 2* — is charge moving in ordinary three-dimensional space, and needs no relativity. The **collective magnetism of many atoms** — real magnets — needs the exchange coupling that a geometry of separate charge territories does not carry.

That is offered honestly, boundary and all, because the boundary is itself the result: it says precisely which part of magnetism is "just charge in motion" and which part is genuinely more. And it leaves a question worth asking out loud:

**If the magnetic moment of an atom, and even the electron's *g = 2*, can be had from charge circulating in real space without relativity — how much of what we call "intrinsic," "quantum," and "relativistic" is actually geometry we hadn't finished reading?**

---

*Full argument, equations, and runnable computations are in "Magnetism in RealQM: Currents, the Nuclear Electron, and the Spin Residue," with the broader programme (RealQM, RealNucleus) and interactive simulations at [claes542.github.io/RealMolecule](https://claes542.github.io/RealMolecule/gallery.html).*
