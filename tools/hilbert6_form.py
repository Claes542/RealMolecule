"""
HILBERT-6, PROGRAMME ITEM 5: verify the FORM of the macroscopic law, not only its coefficient.

THE OBJECTION BEING ANSWERED.  Writing C = (1/V) d^2E/de^2 presupposes that E is quadratic in the
strain near equilibrium -- that is, it ASSUMES Hooke's law rather than deriving it.  Homogenisation
would instead produce linear elasticity as a limit, form and coefficient together, and that is a
real advantage of the formal route.  The programme's answer is that the form can be established
computationally: compute E(e) across a range, show it is quadratic where linear elasticity holds,
and show the departure setting in where nonlinearity is observed.  A demonstration, not a theorem.

TWO TESTS HERE, and the second is the quantitative one.

(1) THE FORM ITSELF, on the 1D periodic chain of tools/hilbert6_elastic.py.  Strain the cell,
    e = (a-a0)/a0, and ask over what range E(e) is quadratic and how large the cubic term is.
    This is a mathematical question about the energy functional and does not need a material.

(2) THE FIRST ANHARMONIC COEFFICIENT, ON REAL METALS, which turns the form check into a
    prediction.  The cubic term of E is not a free shape parameter: it fixes B' = dB/dP, the
    pressure derivative of the bulk modulus, which is measured.  So the departure from quadratic
    is itself falsifiable, and checking it uses nothing beyond the alkali energy curves already
    computed in real_plasma_numerics/metal_alkalis.py.

B' is obtained without fitting any equation of state: from E(V) take
        P = -dE/dV,     B = V d^2E/dV^2,     B' = dB/dP at P = 0,
all by finite differences on the computed curve, which is the same discipline as everywhere else
in this programme -- differentiate the computed number, do not fit a form to it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'real_plasma_numerics'))
import numpy as np
from hilbert6_elastic import solve_chain

np.seterr(all='ignore')


# ------------------------------------------------------------------ (1) the form, on the 1D chain
def chain_form(ncell=41, eps=0.35, M=20, a_ref=3.0, span=0.22, npts=23):
    """E(a) across a range of strain about a reference cell, and how far E stays quadratic.

    NOTE, measured: this 1D chain has NO stress-free state in range -- dE/da > 0 at a=3 and the
    energy keeps falling as a decreases, the softened 1D Coulomb giving no binding minimum.  So it
    is a compression model, and the form is tested about a PRE-STRESSED reference, where Hooke's
    law means that E minus its linear term is quadratic.  The stress-free version of the same test
    is part (2), on the alkali metals, which do have an equilibrium.
    """
    As = a_ref*(1.0 + np.linspace(-span, span, npts))
    Es = np.array([solve_chain(a, n=ncell, eps=eps, M=M)['E'] for a in As])
    a0, E0 = a_ref, float(Es[npts//2])
    e = (As - a0)/a0
    # quadratic fitted on the SMALL-strain window only, then extrapolated outward
    win = np.abs(e) <= 0.05
    c = np.polyfit(e[win], Es[win], 2)        # linear term allowed: the reference is pre-stressed
    quad = np.polyval(c, e)
    C2 = 2*c[0]                            # E ~ E0 + (1/2) C2 e^2
    # cubic coefficient from a wider fit, for the size of the first anharmonic term
    win3 = np.abs(e) <= 0.18
    c3 = np.polyfit(e[win3], Es[win3], 3)
    return dict(As=As, Es=Es, a0=a0, E0=E0, e=e, quad=quad, C2=C2,
                cubic=6*c3[0], ratio=abs(6*c3[0])/abs(C2))


# ------------------------------------------------------------------ (2) B' on the alkali metals
def alkali_Bprime():
    """B and B' = dB/dP at zero pressure, from the computed E(r_s) curves. No EOS is fitted."""
    import metal_alkalis as ma
    out = {}
    for el, (rc, rs_exp, B_exp) in ma.ALKALIS.items():
        f = ma.make_energy(rc)
        rs0 = ma.minimise(lambda x: f(x)[0])            # RealQM (split-domain) column
        # work in volume per ion
        def V(rs): return (4*np.pi/3)*rs**3
        def E(v): return f((3*v/(4*np.pi))**(1.0/3.0))[0]
        v0 = V(rs0)
        d = 0.012*v0
        vs = v0 + d*np.arange(-3, 4)
        Ev = np.array([E(v) for v in vs])
        # P = -dE/dV and B = V d2E/dV2 on the stencil, then B' = dB/dP at the centre
        dE = np.gradient(Ev, d, edge_order=2)
        d2E = np.gradient(dE, d, edge_order=2)
        P = -dE
        B = vs*d2E
        k = 3
        dBdP = np.gradient(B, edge_order=2)/np.gradient(P, edge_order=2)
        out[el] = dict(rs0=rs0, B=B[k]*ma.HA_GPA, Bp=dBdP[k], B_exp=B_exp)
    return out


if __name__ == '__main__':
    print("HILBERT-6 ITEM 5 -- verifying the FORM of the macroscopic law\n")

    print("(1) IS E QUADRATIC IN THE STRAIN?  1D periodic chain, fit on |e|<=0.04, extrapolated out")
    R = chain_form()
    print(f"    reference a0 = {R['a0']:.5f} (pre-stressed),  E0 = {R['E0']:.6f},  C2 = {R['C2']:.6f}")
    print(f"    cubic/quadratic coefficient ratio = {R['ratio']:.3f}\n")
    print("      strain e     E(e)          quadratic     departure    rel to the quadratic term")
    for e, E, q in zip(R['e'], R['Es'], R['quad']):
        qt = 0.5*R['C2']*e*e
        rel = abs(E - q)/abs(qt) if abs(qt) > 1e-12 else float('nan')
        flag = ''
        if abs(e) > 1e-6:
            flag = 'quadratic' if rel < 0.05 else ('5-20%' if rel < 0.20 else 'DEPARTS')
        print(f"    {e:+9.4f} {E:14.7f} {q:14.7f} {E-q:+12.2e} {rel:12.3f}   {flag}")

    print("\n(2) THE FIRST ANHARMONIC COEFFICIENT ON REAL METALS")
    print("    B' = dB/dP at P=0, from the computed E(V); measured values for the alkalis are ~3-4")
    print("     el     r_s0      B (GPa)    B exp     B' (model)")
    for el, d in alkali_Bprime().items():
        print(f"    {el:3s} {d['rs0']:9.3f} {d['B']:10.2f} {d['B_exp']:8.1f} {d['Bp']:13.3f}")
