#!/usr/bin/env python3
"""
Demonstration of Enflo's extremal-vector construction
(per arXiv:2305.15442v2, On the Invariant Subspace Problem in Hilbert Spaces).

We work in a finite-dimensional Hilbert space H = C^n with a bounded operator T
(here a truncated weighted shift, small norm) and run the inner extremal-vector
minimisation that lies at the core of Enflo's Main Construction:

    given y' ∈ H, x0 ∈ H, target ε > 0,
    find ℓ' = (a_0, a_1, ..., a_K) of minimal ℓ²-coefficient norm
    such that ‖x0 − ℓ'(T) y'‖ = ε.

Then we evaluate εθ = ⟨ℓ'(T) y', x0 − ℓ'(T) y'⟩, which Enflo shows is a real
non-negative number, and which the Main Construction drives to zero by
iteratively updating y'. εθ → 0 ⇒ ℓ'_∞(T) y'_∞ is non-cyclic ⇒ a non-trivial
closed invariant subspace exists.

Finite-dim is not a test of the theorem (every n×n matrix trivially has
invariant subspaces from eigenvectors), but it makes the mechanics concrete.
"""

import numpy as np
from scipy.optimize import brentq

# ----------------------------------------------------------------------
# Setup: small bounded operator on H = R^n
# ----------------------------------------------------------------------
n = 12                        # dimension
K = 8                         # max polynomial degree (T^j for j=0..K)
target_eps = 0.5              # 0.3 ≤ ε ≤ 0.7 as in the paper

# Truncated weighted shift T e_i = w_i e_{i+1}.  We pick ‖T‖ moderate so
# V_y = [y, Ty, T^2y, ...] stays well-conditioned for the optimisation.
# (Enflo uses ‖T‖ = 10^-20 to avoid delicate estimates; we don't need that.)
T = np.zeros((n, n))
for i in range(n-1):
    T[i+1, i] = 0.7                                       # ‖T‖ = 0.7

x0 = np.zeros(n); x0[0] = 1.0                              # x0 = e_0

# Initial y' (normalised)
rng = np.random.default_rng(0)
y = rng.standard_normal(n); y /= np.linalg.norm(y)


def build_V(T, y, K):
    """Build the matrix V_y whose columns are T^j y, j=0..K."""
    n = len(y)
    V = np.zeros((n, K+1))
    Tj_y = y.copy()
    V[:, 0] = Tj_y
    for j in range(1, K+1):
        Tj_y = T @ Tj_y
        V[:, j] = Tj_y
    return V


def extremal_polynomial(V, x0, eps):
    """
    Minimise ‖a‖_2 subject to ‖V a − x0‖ = eps.
    Lagrangian:  L = ‖a‖² + λ (‖V a − x0‖² − ε²)
    First-order: (I + λ V^T V) a = λ V^T x0
                 a(λ) = λ (I + λ V^T V)^{-1} V^T x0
    Then choose λ so the constraint holds.
    """
    K1 = V.shape[1]
    VtV = V.T @ V
    Vt_x0 = V.T @ x0

    def a_of_lambda(lam):
        return lam * np.linalg.solve(np.eye(K1) + lam * VtV, Vt_x0)

    def constraint(lam):
        a = a_of_lambda(lam)
        return np.linalg.norm(V @ a - x0) - eps

    # Bracket: small λ → residual ≈ ‖x0‖ (1) ; large λ → 0
    # Find a working bracket
    lam_lo, lam_hi = 1e-8, 1.0
    while constraint(lam_lo) < 0 and lam_lo > 1e-20:
        lam_lo *= 0.1
    while constraint(lam_hi) > 0 and lam_hi < 1e12:
        lam_hi *= 10
    try:
        lam_star = brentq(constraint, lam_lo, lam_hi, xtol=1e-10)
        a_opt    = a_of_lambda(lam_star)
    except ValueError:
        # No bracket found — return the small-λ pseudo-inverse fallback
        a_opt = np.linalg.lstsq(V, x0, rcond=None)[0]
        lam_star = np.nan
    return a_opt, lam_star


def report(V, a, x0):
    Va        = V @ a
    residual  = x0 - Va
    res_norm  = np.linalg.norm(residual)
    a_norm    = np.linalg.norm(a)
    eps_theta = float(np.dot(Va, residual))   # ⟨V a, x0 − V a⟩
    return dict(a_norm=a_norm, res_norm=res_norm, eps_theta=eps_theta,
                Va=Va, residual=residual)


# ----------------------------------------------------------------------
# Type 1 / Type 2 classifier
# ----------------------------------------------------------------------
# Type 1 (eq. 19):  exists u_0 of unit norm such that for every n>=1 there is
#     delta_n > 0 such that for every y with |<y/||y||, u_0>| >= 1/100,
#     there is j >= n with |<T^j y, y>| >= delta_n ||y||^2.
# Type 2 (eq. 20):  the negation -- for every u_0 there is an m>=1 such that
#     for every delta > 0 there is y with |<y/||y||, u_0>| >= 1/100 and
#     |<T^j y, y>| <= delta ||y||^2 for all j >= m.
#
# In finite dim we can test by Monte-Carlo sampling y over the unit sphere
# in the cap of vectors close to u_0.
def classify_type(T, n_max=6, n_samples=200, seed=1):
    """
    Decide Type 1 / Type 2 for the operator T. Returns (type_label, u0, evidence).
    We try u_0 = e_0, e_1, ... and the candidate whose inf-over-y-of-max-over-j
    is largest. If that inf is positive for some j up to n_max we call it Type 1.
    """
    rng = np.random.default_rng(seed)
    d   = T.shape[0]
    best_label = "type-2"
    best_u0    = None
    best_delta = 0.0

    for k in range(min(d, 4)):                  # try a few u0 candidates
        u0 = np.zeros(d); u0[k] = 1.0
        # for many random y in the cap |<y/||y||, u0>| >= 1/100, compute
        # max_{1<=j<=n_max} |<T^j y, y>| / ||y||^2 and take the infimum
        min_over_y = np.inf
        for _ in range(n_samples):
            y = rng.standard_normal(d)
            y = y / np.linalg.norm(y)
            if abs(np.dot(y, u0)) < 1e-2:
                continue
            Tjy = y.copy()
            best_j = 0.0
            for j in range(1, n_max + 1):
                Tjy = T @ Tjy
                v = abs(np.dot(Tjy, y))
                if v > best_j:
                    best_j = v
            if best_j < min_over_y:
                min_over_y = best_j
        if min_over_y > best_delta:
            best_delta = min_over_y
            best_u0    = u0
            best_label = "type-1" if min_over_y > 1e-8 else "type-2"

    return best_label, best_u0, best_delta


# ----------------------------------------------------------------------
# Lemma 2 preparation: rescale y so |a_0|^2 dominates |a_j|^2, j>=1.
# ----------------------------------------------------------------------
# Following the paper (eqs. 22-23):  pick s so y_1 = s * y'_1 with
#   1/(20 * sqrt(eps_theta)) < ||y_1|| < 1/sqrt(eps_theta)
# and re-solve the extremal problem with y_1 in place of y'. The minimal-norm
# polynomial then has dominating a_0 because a_0 * y_1 carries most of x_0 and
# the higher T^j y_1 terms must remain small in the new coefficients.
def lemma2_prepare(T, y, x0, eps, K):
    """Return rescaled y1 = s*y and corresponding extremal a, satisfying
    the dominating-a_0 condition (heuristic)."""
    V = build_V(T, y, K)
    a, _ = extremal_polynomial(V, x0, eps)
    eps_theta = float(np.dot(V @ a, x0 - V @ a))
    if eps_theta <= 0:
        return y, a, eps_theta, 1.0
    # Range for s from eq. 23, applied to current y direction
    s_lo = 1.0 / (20.0 * np.sqrt(eps_theta) * np.linalg.norm(y))
    s_hi = 1.0 / (np.sqrt(eps_theta) * np.linalg.norm(y))
    s    = np.sqrt(s_lo * s_hi)                # geometric mean
    y1   = s * y
    V1   = build_V(T, y1, K)
    a1, _ = extremal_polynomial(V1, x0, eps)
    return y1, a1, eps_theta, s


def main():
    global y
    print(f"n = {n}, K = {K}, target ε = {target_eps}")
    print(f"‖T‖_op = {np.linalg.norm(T, ord=2):.4f}")
    print()

    # --- (b) Type 1 / Type 2 classification --------------------------
    label, u0, evidence = classify_type(T)
    print(f"=== (b) Operator classification ===")
    print(f"   Type: {label}")
    print(f"   u_0:  e_{int(np.argmax(np.abs(u0)))}")
    print(f"   inf_y max_j |<T^j y, y>|/‖y‖² = {evidence:.4e}")
    print(f"   (positive ⇒ Type 1, MC applies; near zero ⇒ Type 2, eq. (12) needed.)")
    print()

    # --- (a) Per-iteration extremal-vector machinery ------------------
    print(f"=== (a) Extremal-vector solve on a sequence of y' ===")
    print(f"{'iter':>4} {'‖a‖':>10} {'‖x0−Va‖':>12} {'εθ':>12} {'a₀²/Σa²':>10}")

    history = []
    rng_inner = np.random.default_rng(0)
    for it in range(8):
        # Sample a new y' in a controlled way: random with a small cyclic offset
        ynew = rng_inner.standard_normal(n)
        ynew[0] += 0.3 * (it + 1)             # make y' progressively more cyclic
        ynew /= np.linalg.norm(ynew)

        V = build_V(T, ynew, K)
        a, lam = extremal_polynomial(V, x0, target_eps)
        r = report(V, a, x0)
        a0_dom = a[0]**2 / (np.sum(a**2) + 1e-30)
        history.append(r['eps_theta'])
        print(f"{it:>4d}  {r['a_norm']:>10.4f}  {r['res_norm']:>12.6f}  "
              f"{r['eps_theta']:>12.6f}  {a0_dom:>10.4f}")

        # --- Lemma 2 preparation: rescale to dominating a_0 ----------
        if r['eps_theta'] > 1e-6 and a0_dom < 0.99:
            y1, a1, eth, s = lemma2_prepare(T, ynew, x0, target_eps, K)
            V1 = build_V(T, y1, K)
            r1 = report(V1, a1, x0)
            a0_dom_new = a1[0]**2 / (np.sum(a1**2) + 1e-30)
            print(f"      → Lemma 2: s={s:.3e}, a₀²/Σa² {a0_dom:.4f} → {a0_dom_new:.4f}, "
                  f"εθ {r['eps_theta']:.4f} → {r1['eps_theta']:.4f}")

    print()
    print("=== Summary ===")
    print(f"   εθ trajectory: {['%.4f' % v for v in history]}")
    if label == "type-1":
        print("   Type 1 operator: MC iteration applies in principle; full")
        print("   Fourier-analytic update (eq. 13) would drive εθ → 0.")
    else:
        print("   Type 2 operator: MC alone won't converge; Enflo's eq. (12)")
        print("   construction is invoked to find a non-cyclic vector directly.")


if __name__ == "__main__":
    main()
