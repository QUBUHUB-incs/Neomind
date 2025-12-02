# SageMath code — paste into Sage / SageMathCell

# Define the quartic curve
R.<x,y> = QQ[]
f = 12*x^4 + 16*x^3 - 59*x^2 - 14*x + 49
C = Curve(y^2 - f)   # plane affine curve y^2 = f(x)

# Known rational points
A = (-3, 10)
B = (-1, 0)
Cpt = (0, 7)    # rename Cpt to avoid clash with curve C
D = (1, 2)

# Utility: solve linear system to get parabola y = A x^2 + B x + C through three points
def parabola_through(p0, p1, p2):
    x0,y0 = QQ(p0[0]), QQ(p0[1])
    x1,y1 = QQ(p1[0]), QQ(p1[1])
    x2,y2 = QQ(p2[0]), QQ(p2[1])
    M = matrix(QQ, [[x0^2, x0, 1],
                    [x1^2, x1, 1],
                    [x2^2, x2, 1]])
    v = vector(QQ, [y0, y1, y2])
    sol = M.solve_right(v)   # returns vector [A,B,C]
    return tuple(sol)        # (A,B,C) rational

# Given parabola y = A x^2 + B x + C, compute intersections with quartic:
def parabola_intersections(par):
    Acoef,Bcoef,Ccoef = par
    # build polynomial (A x^2 + B x + C)^2 - f(x)
    P = (Acoef*x^2 + Bcoef*x + Ccoef)^2 - f
    # factor and get roots
    P = P.expand()
    fac = P.factor()
    # return list of roots (with multiplicity) in algebraic closure if rational
    roots = P.roots(ring=QQbar)
    return roots, P

# Given three rational pts p,q,r, return the fourth intersection point (x,y)
def Padd(p,q,r):
    par = parabola_through(p,q,r)
    roots, Ppoly = parabola_intersections(par)
    # roots is list of (root, multiplicity) as algebraic numbers. We want the rational root
    # The x-coordinates of p,q,r are roots; find the remaining root.
    xs = set([QQ(p[0]), QQ(q[0]), QQ(r[0])])
    # Factor Ppoly over QQ to try to find rational factors:
    Prat = Ppoly.factor()
    # Convert to polynomial over QQ and divide by (x-x0)(x-x1)(x-x2)
    denom = (x - QQ(p[0]))*(x - QQ(q[0]))*(x - QQ(r[0]))
    other = (Ppoly / denom).simplify_full()
    # other should be linear; solve for its root
    # We solve Ppoly = 0 and pick the x different from the known three
    sols = [s[0] for s in Ppoly.roots(QQbar)]
    # Pick the root not among the three xs (numerical algebraic comparison)
    for sol in sols:
        if not any(sol == xi for xi in xs):
            x_new = sol
            break
    else:
        # fallback — if multiplicities / algebraic, approximate and compare numerically
        sol_approx = [s for s in sols if not any(abs(complex(s) - complex(xi)) < 1e-10 for xi in xs)]
        x_new = sol_approx[0]

    # compute y from parabola
    Acoef,Bcoef,Ccoef = par
    y_new = Acoef*x_new^2 + Bcoef*x_new + Ccoef
    # Convert to QQ if possible
    x_new = QQ(x_new)
    y_new = QQ(y_new)
    return (x_new, y_new)

# Map a quartic point (x,y) to (a,b,c)
def point_to_abc(pt):
    xv = QQ(pt[0]); yv = QQ(pt[1])
    num = 6*xv^2 - 13*xv + 7
    den = 2*(6*xv - 7)
    a_rat = (num + yv) / den
    b_rat = (num - yv) / den
    # clear denominators to get integers
    mult = lcm(a_rat.denominator(), b_rat.denominator(), 1)
    Aint = a_rat * mult
    Bint = b_rat * mult
    Cint = mult
    return (Aint, Bint, Cint)

# Recreate A..J as in your notes
Apt = A
Bpt = B
Cpt = Cpt
Dpt = D

def safe_p(*pts):
    return Padd(*pts)

E = safe_p(Apt, Bpt, Cpt)    # E = P(A,B,C)
F = safe_p(Bpt, Cpt, Dpt)
G = safe_p(Apt, Cpt, Dpt)
H = safe_p((-Apt[0], -Apt[1]), Bpt, Dpt)   # -A is negation of y
I = safe_p(Apt, Bpt, Dpt)
J = safe_p((-Apt[0], -Apt[1]), Bpt, Cpt)

# Show points and map one to (a,b,c) to confirm
print("A,B,C,D:", Apt, Bpt, Cpt, Dpt)
print("E,F,G,H,I,J:", E, F, G, H, I, J)
print("Example map E -> (a,b,c):", point_to_abc(E))

# You can generate K..V by following the same pattern in your narrative,
# e.g. K = P(-A,B,E) etc. Use find_point logic similar to your code to iterate.
