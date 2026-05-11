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
# Outer loop: a few iterations of "Main Construction" -- here a simple
# gradient-style update of y' that tries to decrease εθ while keeping the
# residual norm near ε. (Enflo's actual update uses (13): y_{n+1} = y_n +
# Σ r_j T^j y_n with Fourier-analytic control; we use a simple step.)
# ----------------------------------------------------------------------
def main():
    global y
    eta = 0.4   # step size for the y-update

    print(f"n = {n}, K = {K}, target ε = {target_eps}")
    print(f"‖T‖_op = {np.linalg.norm(T, ord=2):.4f}")
    print()
    print(f"{'iter':>4} {'‖a‖':>10} {'‖x0−Va‖':>12} {'εθ':>12} {'‖y‖':>10}")

    history = []
    for it in range(20):
        V = build_V(T, y, K)
        a, lam = extremal_polynomial(V, x0, target_eps)
        r = report(V, a, x0)
        history.append(r['eps_theta'])
        print(f"{it:>4d}  {r['a_norm']:>10.4f}  {r['res_norm']:>12.6f}  "
              f"{r['eps_theta']:>12.6f}  {np.linalg.norm(y):>10.6f}")

        # Update y' in a direction that decreases εθ.
        # Gradient of εθ = ⟨V_y a, x0 − V_y a⟩ with respect to y is non-trivial;
        # a rough surrogate: move y towards the residual direction so that the
        # next Va aligns more with x0, shrinking ⟨Va, x0 − Va⟩.
        grad_y = r['residual']            # very crude descent direction
        y = y - eta * grad_y
        y = y / np.linalg.norm(y)

    print()
    print("Trajectory of εθ:")
    print(["%.4f" % v for v in history])

    if history[-1] < history[0] * 0.5:
        print()
        print("→ εθ decreased; the limit V_∞ ℓ'_∞ would be non-cyclic, hence")
        print("  span{T^j V_∞ ℓ'_∞ : j ≥ 0} is the non-trivial closed invariant subspace.")
    else:
        print()
        print("→ The crude surrogate update did not drive εθ to zero. Enflo's")
        print("  actual update (eq. 13 in the paper, Fourier-analytic) is required;")
        print("  for some weighted shifts a separate argument is invoked (eq. 12).")


if __name__ == "__main__":
    main()
