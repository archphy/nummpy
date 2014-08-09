from aux import d_n, maxmin_fx, max_
import math
import numpy

def chebyshev(f, n):
	"""Returns the n Chebyshev nodes (x, f(x)) from -1 to 1."""

	xs, ys = [], []
	for k in xrange(n):
		xs.append(math.cos((((2*k)+1.0)/(2*(n+1.0)))*math.pi))
		ys.append(f(xs[-1]))
	return (xs, ys)

def lagrange_pol(x, j, k, xs):
	"""jth Lagrange Polynomial at x with degree k.
	
	x: The point to evaluate the polynomial at. (float or sympy.symbols)
	j: The jth Lagrange Polynomial. (int)	
	k: The polynomial's degree. (int)	
	xs: The base points. (set of floats)
	
	Use lagrange_interpolation to obtain the Lagrange interpolation polynomial.
	"""
	
	m = 0
	terms = []
	while m <= k:
		if m != j:
			terms.append((x - xs[m]) / (xs[j] - xs[m]))
		m += 1
	return numpy.prod(terms)

# This isn't intended to be efficient, but to be clear what is being done and 
# to separate the Lagrange Polynomial generation from the interpolation itself.
# For efficiency, something like scipy's Lagrange interpolation algorithm
# github.com/scipy/scipy/blob/v0.14.0/scipy/interpolate/interpolate.py#L44
# is better. 
def lagrange(x, xs, ys, k=None):
	"""Interpolates (xs, ys) at x via Lagrange.
	
	x: The point to evaluate the polynomial at. (float or sympy.symbols)
	xs: The x points. (set of floats)
	ys: The y points. (set de floats)
	k: Use a degree k poly. If not supplied, adjust accordingly. (int)
	
	E.g. lagrange_interpolation(0.5, [1.0, 2.0, 3.0], [1.0, 4.0, 9.0])
	"""
	
	if k is None:
		k = len(xs) - 1
	i = 0
	terms = []
	while i <= k:
		terms.append(lagrange_pol(x, i, k, xs)*ys[i])
		i += 1
	return sum(terms)

def divided_differences(xs, ys, xs_=None):
	"""Divided differences f[x0,...,xn], onde f[x0]=y0,...,f[xn]=yn.
	
	xs: The xs from f[x0,...,xn]. (set of floats)
	ys: The ys from f[x0],...,f[xn]. (set of floats)		
	xs_: Do not supply. (None)
	
	Use newton_interpolation to obtain the interpolation polynomial.	
	"""
	
	if xs_ is None:
		xs_ = xs
	if len(xs_) > 1:
		return ((divided_differences(xs, ys, xs_[1:]) - (
				divided_differences(xs, ys, xs_[:-1]))) / (xs_[-1] - xs_[0]))
	elif len(xs_) == 1:
		return ys[xs.index(xs_[0])]
	else:
		raise ValueError("xs' dimension must be >= 1.")

def newton(x, xs, ys):
	"""Interpolates (xs, ys) at x via Newton (divided differences).
	
	x: The point to evaluate the polynomial at. (float or sympy.symbols)
	xs: The x points. (set of floats)
	ys: The y points. (set de floats)
	
	E.g.
	newton_interpolation(
		sympy.symbols('x'), [1.0, 2.0, 3.0], [1.0, 4.0, 9.0]
	)		
	"""
	
	result = ys[0]
	for i in xrange(2, len(xs)+1):
		dd = divided_differences(xs[:i], ys)
		pol = [(x-xn) for xn in xs[:i-1]]
		for j in pol:
			dd *= j
		result += dd
	return result

# TODO: Review and test
# def maxerror_h(f_s, xs):
# 	"""Theoritical maximum error of a interpolation of equally-spaced points.
# 	
# 	f_s: The function being interpolated. (python-compatible function)	
# 	xs: The x points used in the interpolation. (set of floats)
# 	"""
# 	
# 	n = len(xs)
# 	h = (xs[-1] - xs[0]) / (n-1)
# 	d_n_ = d_n(f_s, n)		# If an error occurs, it'll be raised here
# 	max_d_n_ = maxmin_fx(d_n_, xs, max_)[1]
# 	return ((h**n)/(4.0*n))*abs(max_d_n_)