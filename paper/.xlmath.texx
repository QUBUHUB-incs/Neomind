from sympy import symbols, Poly, solve, simplify, Rational, expand
from math import lcm

def find_third_point(a1, b1, a2, b2):
    """
    Find the third intersection point of a line through points (a1, b1) and (a2, b2)
    on the cubic curve a(a+1)(a+b) + b(b+1)(a+b) + (a+1)(b+1) = 4(a+1)(b+1)(a+b).
    
    Args:
        a1, b1: Coordinates of the first point (rational numbers)
        a2, b2: Coordinates of the second point (rational numbers)
    
    Returns:
        Tuple (a3, b3): Coordinates of the third intersection point
    """
    # Define symbolic variables
    t, a, b = symbols('t a b')
    
    # Parameterize the line: a = a1 + t*(a2 - a1), b = b1 + t*(b2 - b1)
    a_expr = a1 + t * (a2 - a1)
    b_expr = b1 + t * (b2 - b1)
    
    # Define the cubic curve equation:
    # a(a+1)(a+b) + b(b+1)(a+b) + (a+1)(b+1) = 4(a+1)(b+1)(a+b)
    left_side = a * (a + 1) * (a + b) + b * (b + 1) * (a + b) + (a + 1) * (b + 1)
    right_side = 4 * (a + 1) * (b + 1) * (a + b)
    equation = left_side - right_side
    
    # Substitute the line parameterization into the equation
    poly_t = equation.subs({a: a_expr, b: b_expr})
    
    # Simplify and convert to a polynomial in t
    poly_t = expand(poly_t)
    poly = Poly(poly_t, t)
    
    # Get the coefficients of the cubic polynomial
    coeffs = poly.coeffs()
    
    # The polynomial is cubic (degree 3), so coeffs = [c3, c2, c1, c0]
    # For a cubic c3*t3 + c2*t2 + c1*t + c0, the sum of roots is -c2/c3
    while len(coeffs) < 4:
        coeffs.append(0)
    if len(coeffs) != 4:
        raise ValueError("Unexpected polynomial degree. Expected cubic polynomial.")
    
    c3, c2, c1, c0 = coeffs
    
    # The known roots are t=0 (for point 1) and t=1 (for point 2)
    # Sum of roots: t1 + t2 + t3 = -c2/c3
    # Since t1 = 0, t2 = 1, we have 0 + 1 + t3 = -c2/c3
    t3 = -c2 / c3 - (0 + 1)
    
    # Compute the third point coordinates
    a3 = a1 + t3 * (a2 - a1)
    b3 = b1 + t3 * (b2 - b1)
    
    # Simplify the results to ensure rational output
    a3 = simplify(a3)
    b3 = simplify(b3)
    
    return (a3, b3)

# Verify the a point is on the curve
def is_on_curve(a, b):
    left = a * (a + 1) * (a + b) + b * (b + 1) * (a + b) + (a + 1) * (b + 1)
    right = 4 * (a + 1) * (b + 1) * (a + b)
    return left == right

def is_too_big(x, y):
    xn, xd = x.as_numer_denom()
    yn, yd = y.as_numer_denom()
    return max(abs(xn), abs(xd), abs(yn), abs(yd)) > 10**200

def fin_smallest_all_positive_point():
    points = [(Rational(-11, 1), Rational(-4, 1)), (Rational(11, 9), Rational(-5, 9))]
    queue = [(points[0], points[1])]

    def accept(x, y):
        print(x, y)
        for (x2, y2) in points:
            queue.append(((x2, y2), (x,y)))
        points.append((x, y))

    while len(queue) > 0:
        print(len(queue))
        (x1, y1), (x2, y2) = queue.pop()
        x3, y3 = find_third_point(x1, y1, x2, y2)
        if not is_too_big(x3, y3):
            if (x3, y3) not in points:
                accept(x3, y3)
            if (y3, x3) not in points:
                accept(y3, x3)

    print(points)
    eligible = [(x,y) for (x,y) in points if x > 0 and y > 0]
    return min(
        eligible,
        key=lambda x: lcm(x[0].as_numer_denom()[1], x[1].as_numer_denom()[1])
    )
