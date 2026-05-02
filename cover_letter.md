# Cover letter — Foundations of Physics submission

To the Editor-in-Chief,
*Foundations of Physics*

Dear Editor,

I am submitting for your consideration the manuscript

> **Quantum Mechanics as Multiphase 3D Continuum Mechanics — Realised through Mind–AI Cooperation**

for publication in *Foundations of Physics*. The paper presents a real-space wave-function reformulation of the *N*-electron quantum-mechanical problem in which the wave function $\Psi(\mathbf{x}) = \sum_{i=1}^N \psi_i(\mathbf{x})$ lives on physical $\mathbb{R}^3$ as a sum of one-electron wave functions $\psi_i$ on non-overlapping spatial domains $\Omega_i$ meeting at a Bernoulli free boundary. Each $\psi_i^2$ is the actual electron charge density of electron $i$ in its own region of space. Geometry, bonding, dynamics, reactive chemistry, protein folding, and condensed-phase melting all emerge from a single variational principle, with no chemistry-specific empirical input beyond kernel charges and the computational mesh size. The framework returns to the real-space density picture Schrödinger originally proposed in his 1926 letter to Lorentz — a picture abandoned during the late 1920s under the dominance of the Bohr–Heisenberg–Dirac Copenhagen interpretation.

The paper makes a foundational claim. The hundred-year question of whether chemistry reduces to quantum physics — Dirac's 1929 position vs. the autonomy view defended by leading chemists for nearly a century — has been driven less by disagreement on principle than by the practical impossibility of solving the standard $3N$-dimensional wave-function equations without chemistry-specific approximations whose presence vindicates the autonomy view. The reformulation presented here recovers atom energies (Li–Rn within ~1%, including a heuristic Periodic-Table aufbau from minimum-energy electron packing), the H₂ covalent bond (within 3% of Kolos–Wolniewicz), atomization energies of closed-shell hydrides across four periodic-table groups, S66 dimer geometries, proton-transfer reactions, protein folding, excited states, and a three-system condensed-phase calibration (NaCl, CO₂ dry ice, solid N₂) — all through Coulomb interactions in real 3D space. **Stability of matter** for $N$ atoms with kernel charge $Z$ follows from the Hardy inequality applied to the global wave function as a one-paragraph derivation, giving $E \ge -C N Z^3$ — in striking contrast to the heroic Dyson–Lenard–Lieb–Thirring proof required in standard quantum mechanics. The same multiphase formulation extends naturally to atomic nuclei in a proton–electron picture (4 protons + 2 electrons for He-4), demonstrating reach beyond chemistry to nuclear-scale physics. The conclusion the paper draws is concrete: **Dirac's position was correct, but realised outside the 1929 wave-function-on-$3N$-space formulation that he himself proposed**.

I believe the manuscript is well suited to *Foundations of Physics* on several grounds:

1. The chemistry-vs-physics framing is foundational rather than methods-driven and does not fit cleanly into a chemistry-methods journal.
2. The variational principle is reformulated as a Bernoulli free-boundary problem in 3D real space — a novel mathematical structure with consequences for regularity theory, level-set methods, and Dirichlet–Neumann decompositions.
3. The implementation runs interactively in a browser on a consumer GPU at $10^2$–$10^4 \times$ standard quantum-chemistry speeds, lowering the barrier to inspection and reproduction. A curated public Gallery (linked in Section 8) provides interactive visualisations of every result reported in the paper.
4. The manuscript also reports a worked case of how a single mathematical mind, paired with an AI engineering partner (Claude, Anthropic), produced research-grade interactive scientific software at a scale that previously required a programming team. We document the collaboration honestly, including the corrections and retractions that occurred during its course, because we believe such cases will become increasingly relevant to the practice of theoretical physics.

The work is original and has not been submitted elsewhere. The author is sole and has no competing interests to declare. The full source code, validation suite, and interactive Gallery are publicly available at the URLs in Section 8 of the manuscript.

I would be happy to provide any further information the editorial process requires. I appreciate your consideration.

Sincerely,

**Claes Johnson**
KTH Royal Institute of Technology, Stockholm, Sweden
claesjohnson@gmail.com
2 May 2026

---

*Enclosures*: manuscript PDF (`RealQMarXiv4.pdf`), LaTeX source (`RealQMarXiv4.tex`), figure (`figures/validation_summary.pdf`), arXiv preprint identifier (when available).
