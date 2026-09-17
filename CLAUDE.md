# Working rules for this repo

## Check before building

**Before implementing any new solver mechanism or driver page, search for it first and say in the
reply what was found.** Read `~/.claude/projects/-Users-cgjoh-Development-H2O/memory/project_capability_map.md` — it lists
solver features by what they *do* and names the page implementing each system. Most things here
already exist.

- New mechanism → `grep -rn "<concept>" *.js` and `grep -l "<concept>" *.html` first.
- New system (a molecule, a scan) → `ls *<species>*` first. Start from a page that runs.
- When Claes says "we must have done this before" or "we have done this 1000 times", that is
  almost certainly correct. Go and look. Do not answer from recall.
- He should never be the mechanism by which prior work is remembered.

On 2026-09-17 three mechanisms were rebuilt from scratch that already existed
(`USER_CONST_ECORE`+`USER_FREEZE_ECORE`, `mol_fast_H3X_split_scan.html`, `USER_FULL_RC`), each
caught only by him asking. That cost most of a day.

## Before quoting any number

- **Step ladder.** 4000 steps is not converged: ~11 kcal/mol drift per 50 steps. Run 4k/8k/16k/32k
  and show the difference stabilising. Differences below ~10 kcal/mol from a 4000-step run carry no
  information.
- **Mesh ladder** for any small energy difference.
- **Check the geometry that was actually built** — print the realised bond lengths and angles.
  Kernel positions must not be rounded to integer cells (worth 39.5 kcal/mol in NH3, and it
  destroys T_d).
- **Suspect the control before redesigning the model.** A failed symmetry control was twice an
  instrumentation bug, and once drove a model redesign that was not needed.
- Never retype a number between messages; recompute or grep it.

## Physics conventions

- Splitting a −Z valence into Z **one-electron** domains is the house construction: n unit charges
  give n(n−1)/2 pairs, each counted once by the i≠j rule — exact, no (Z−1)/Z factor, no kinetic
  fudge. A −Z multi-occupancy blob is the exception and needs justifying.
- Molecules impose only the **inner** cutoff r_c. Outer boundaries are free, found by non-overlap.
  A uniform density plus a free boundary is unstable — it expands without bound.
- Differences within one composition cancel their errors; differences across compositions do not.

## Git

- Do **not** add a `Co-Authored-By: Claude` trailer to commits in this repo.
- Stage only the files actually changed — the working tree carries ~800 unrelated dirty files.
