# Episode 1 — "Calculemus"

Series frame: **Leibniz as host.** A static public-domain portrait (Francke, 1695) on the open
and the close only — never over the simulations. No animated mouth; the screen is the content.

Target 7 minutes. Voiceover over screen capture; no presenter on camera.
**Every number below is on the verified list. Nothing here depends on He = −2.9036 or H₂ = −1.17.**

Pronunciation notes for TTS (respell in the spoken file, keep the real spelling here):
| written | speak as |
|---|---|
| Hartree | HAR-tree |
| Schrödinger | SHUR-ding-er |
| Koopmans | KOHP-mans |
| caesium | SEE-zee-um |
| nuclei | NEW-klee-eye |
| au / a.u. | atomic units |
| 10³ | ten to the three |
| Leibniz | LYBE-nitz |
| calculemus | kal-koo-LAY-mus |

> **Source check before publication:** the Leibniz passage is quoted below from reference
> knowledge. The usual citation is *The Art of Discovery*, 1685. Verify the wording and the date
> against a primary source or a standard edition before this goes out — a misquotation in the
> cold open of episode one would be the worst possible place for one.

---

**[SCREEN: the Francke portrait of Leibniz, still, held]**

In sixteen eighty-five, Leibniz wrote that if two philosophers found themselves in disagreement,
there should be no more need of argument between them than between two accountants. They should
take up their pencils, and say to one another: calculemus. Let us calculate.

He was describing a machine he never built, and a language he never finished. But the instinct
is the one this series is about, and it turns out to be sharper than it looks. Of two accounts of
matter, one can be calculated and one cannot. That does not make the second false. It makes it
unanswerable. And those are different things.

**[SCREEN: portrait fades out. Then `realqm_explainer_hydrogen.html` loading]**

So — calculemus. Here is a hydrogen atom. One proton, one electron.

I am going to compute its energy, and I want you to notice what I do *not* type in. I give the
program one number: the charge on the nucleus. One. That is the whole input. No basis set, no
functional, no fitted parameter, nothing taken from an experiment.

**[SCREEN: the solve running, energy converging]**

Minus one half. In atomic units, the ground state energy of hydrogen is minus one half exactly,
and that is what comes out. Thirteen point six electron volts, if you prefer that unit.

That is not impressive on its own. Hydrogen has been solvable with pencil and paper since
nineteen twenty-six. I show it because it is the anchor: if a model gets hydrogen wrong, nothing
else it says matters. And because of what happens next.

**[SCREEN: `atom_simulator.html`, cursor over the "Periodic table 2..18" button]**

This is the same program. Same equations, same one input per atom — the nuclear charge. I am
going to press one button and it will compute every atom from helium to argon, in order, and
draw the electron shells as they form.

**[SCREEN: press. Let it run. Let the densities build.]**

Watch the shells appear. Nobody put them there. There is no shell structure in the equations —
there is charge, there is Coulomb repulsion, and there is the requirement that each electron's
charge cloud occupy its own region without overlapping its neighbours. The shells are what that
produces.

I want to be careful here, because this is the kind of claim that is easy to overstate. The
*number* of electrons in each shell is supplied — I tell the program how to distribute them. What
is not supplied is that they arrange into shells at all, or where the boundaries between them
fall. Those come out of the minimisation. The honest version of the claim is narrower than "the
periodic table from nothing", and it is still worth watching.

**[SCREEN: the totals table filling in]**

And the energies. Helium, two per cent. Lithium, under one per cent. Boron, two tenths of a per
cent. Across the first three periods the total energies come out within one to two per cent on
average, worst case four and a half per cent at silicon.

Now. One to two per cent is not spectacular accuracy. Standard quantum chemistry does far better
than that on these atoms — for helium it does *twenty digits*. So why would anyone care about a
method that manages two?

**[SCREEN: split — left, the atom table still running; right, a blank panel]**

Because of what happens to those twenty digits when you add electrons.

The Schrödinger equation for N electrons lives in three N dimensions. Two electrons, six
dimensions. Ten electrons, thirty. A hundred electrons, three hundred dimensions. The cost of
solving it directly grows exponentially, and it stops being possible at about three or four
electrons. Everything beyond that — every number standard quantum chemistry produces for a real
molecule — comes from an approximation, and the accuracy you can state depends on which
approximation you chose.

For helium, twenty digits. For a protein: nothing. Not a worse number — no number. You cannot
state an error bar on a calculation nobody can do.

**[SCREEN: the finished atom table]**

This model lives in three dimensions, whatever the electron count. Each electron is a charge
cloud in ordinary space. The cost grows linearly with the number of electrons, not
exponentially. So the accuracy is a few per cent — and it is a few per cent for helium, a few
per cent for argon, and a few per cent for a protein of a hundred and fifty-three residues
running on a laptop.

That is the trade this series is about. Not "more accurate" — less accurate, at the small end,
by a wide margin. But computable all the way up, and therefore answerable all the way up.

**[SCREEN: hydrogen result again, then the gallery page]**

So: hydrogen exactly, with one number supplied. The periodic table to a few per cent, with one
number per atom supplied. Both of those you just watched happen, in a browser, in about a
minute.

Everything in this series runs in a browser. The pages are linked below. If you think a number
is wrong, you can open the page and check it — which is the only reason to believe any of it.

**[SCREEN: the Francke portrait returns, still]**

Leibniz wanted a calculus for settling arguments. What he would have noticed about this one, I
think, is not that it is accurate. It is that you can run it.

Next time: what happens with two electrons, and why that is where the whole difficulty lives.

**[END]**

---

## What this episode deliberately does not say

- It does not claim helium's total energy is exact. It quotes two per cent, which is Table 1.
- It does not mention H₂ at all. That is episode 4, with the bond length.
- It does not say "standard quantum mechanics is wrong." It says it is uncomputable past a few
  electrons, which is a statement nobody disputes.
- It does not use the word "theory".
