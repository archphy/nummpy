# Compare nummpy results with scipy/numpy's ones.
# Only some of the tests are randomized, most are done with pre-defined 
# functions and none cover all possible cases (yet).
# For now, these tests only serve the purpose of making sure changes to the 
# code don't screw things up.
# This software comes with ABSOLUTELY NO WARRANTY.

# TODO: Randomized, case testing

import numpy
import sympy
from scipy import optimize, interpolate, integrate
from aux import tol_iter, d_n
import rootfinding
import interpol
import datafit
import numint
# import linalg

def t_rootfinding():
	
	print 'rootfinding...'
	
	def test_it(expected, result, max_error):
		
		diff = abs(expected-result)
		if diff < max_error:
			print 'OK'
		else:
			print 'FAIL'
			print '		Expected:', expected
			print '		Got:', result
			print '		Difference:', diff
			print '		Tolerance:', max_error
	
		
	def rp(x, k=1, min=-1, max=1):
		"""Random polynomial of degree k at x"""
		
		coeffs = numpy.random.uniform(min, max, size=k+1)
		return sum([coeffs[i]*x**i for i in xrange(len(coeffs))])
	
	x = sympy.symbols('x')
	
	# Number of tests
	n = 4
	
	# Maximum iterations
	iter = 10
	
	# How much can the tests differ from the expected results? Adjust 
	# according to the method and number of iterations.
	max_error = 10**-8
	
	# For simplicity, test only for degree 1 polynomials with roots in a known
	# interval
	print '	Newton...'
	i = 0
	while i < n:
		p = sympy.lambdify(x, rp(x, 1))
		guess = numpy.random.randint(-10, 10)
		dx = d_n(p, 1)
		if dx(x) != 0:
			try:
				expected = optimize.newton(p, guess, dx, maxiter=iter)
			except RuntimeError:	# Doesn't return if approximation
				continue
			print '		Test {}/{}'.format(i+1, n),
			result = rootfinding.newton(p, guess, tol_iter, n=iter)
			test_it(expected, result, max_error)
			i += 1
	
	# The secant method as isn't able to stop if the difference between the 
	# guesses is less than tolerance, whereas scipy's does:
	# github.com/scipy/scipy/blob/v0.14.0/scipy/optimize/zeros.py#L45
	# This (important) feature (and the tests) will soon be implemented 
	print '	Secant... SKIP'
	
	print '	Bisection...'
	
	# Slow method
	iter = 10**2
	
	i = 0
	while i < n:
		p = sympy.lambdify(x, rp(x, 1))
		try:
			expected = optimize.bisect(p, -10, 10, maxiter=iter)
		except RuntimeError:	# Doesn't return if approximation
			continue
		except ValueError:		# Sem raizes no intervalo (um coeff = 0)
			continue
		print '		Test {}/{}'.format(i+1, n),
		result = rootfinding.bisec(p, -10, 10, tol_iter, n=iter)
		test_it(expected, result, max_error)
		i += 1
		

# Only Lagrange is tested since there isn't any divided differences 
# implementation on numpy nor scipy nor sympy.
def t_interpol():
	
	print 'interpol...'
	
	def test_it(expected, result, max_error):
		
		max_diff = max(abs(expected-result))
		if max_diff < max_error:
			print 'OK'
		else:
			print 'FAIL'
			print '		Expected:', expected
			print '		Got:', result
			print '		Difference:', max_diff
			print '		Tolerance:', max_error
	
	# How much can the tests differ from the expected results? Adjust 
	# according to the method and data set used.
	max_error = 10**-6
	
	print '	Lagrange...',
	
	# The function used in the testing, change it if you want
	def f(x):		
		return numpy.sin(2*numpy.pi*x)+numpy.cos(3*numpy.pi*x)
	
	xs = numpy.arange(-1, 1, 10**-1)
	ys = f(xs)
	
	expected = numpy.array(interpolate.lagrange(xs, ys).c[::-1])
	x = sympy.symbols('x')
	result_ = sympy.simplify(interpol.lagrange(x, xs, ys))
	result = []
	for i in xrange(len(expected)):
		result.append(result_.coeff(x, i))
	result = numpy.array(result)
	
	test_it(expected, result, max_error)

def t_datafit():
	
	print 'datafit...'
	
	def test_it(expected, results, max_error, rs=True):
		"""Prints the test results for polyfit."""
		
		print '		coefficients...',
		wanted = expected[0]
		got = results[1]
		max_diff = max(abs(wanted-got))
		if max_diff < max_error:
			print 'OK'
		else:
			print 'FAIL'
			print
			print '		Expected:', wanted
			print
			print '		Got:', got
			print
			print '		Difference:', max_diff
			print '		Tolerance:', max_error

		if rs:
			print '		rs...',
			wanted = expected[1][0]
			got = results[2]
			diff = abs(wanted - got)
			if diff < max_error:
				print 'OK'
			else:
				print 'FAIL'
				print '		Expected:', wanted
				print '		Got:', got
				print '		Difference:', diff
				print '		Tolerance:', max_error
		else:
			print '		rs... SKIP'
	
	# How much can the tests differ from the expected results? Adjust 
	# according to the method and step size used.
	max_error = 10**-5
	
	# Step size used in the sampling
	data_h = 10**-5		# (2*10**5 datapoints with the default test function)
	
	# Step size and method for the numerical integration
	met = 'boole'
	h = 10**-4
	
	# Polynomial degree to use (change accordingly to the function you use)
	n = 5
				
	print '	linear...'
	
	# Simple test function for linear	
	def f(x):
		return x**2
	
	# The test data-set
	xs = numpy.arange(0, 1, data_h)
	ys = f(xs)
	
	# Expected results (using the data-set above)
	expected = numpy.polyfit(xs, ys, 1, full=True)
	
	# Ours
	results = datafit.linear(xs=xs, ys=ys, full=True)
	test_it(expected=expected, results=results, max_error=max_error)
	
	print '	polyfit...'
	
	# The function used in the testing, change it if you want
	def f(x):		
		return sympy.sin(2*sympy.pi*x)+sympy.cos(3*sympy.pi*x)
	
	# Using numpy library
	def fn(x):		
		return numpy.sin(2*numpy.pi*x)+numpy.cos(3*numpy.pi*x)
	
	# The test data-set
	xs = numpy.arange(-1, 1, data_h)
	ys = fn(xs)
	
	# Expected results (using the data-set above)
	expected = numpy.polyfit(xs, ys, n, full=True)
	
	# Ours
	results = datafit.polyfit(xs=xs, ys=ys, n=5, full=True)
	test_it(expected=expected, results=results, max_error=max_error)
	
	print '	polyfitc...'
	
	# It makes sense to increase the maximum difference between numpy's result 
	# (discrete approach) and our continuous approach.
	max_error = 10**-3
	
	# Our results
	results = datafit.polyfitc(f=f, xo=-1, xn=1, n=5, met=met, h=h, full=True)
	# Can't compare the residue calculated via the integral with the one 
	# calculated via the sum of the squares
	test_it(expected=expected, results=results, max_error=max_error, rs=False)
	
	print '	funfit...'	
	
	# Lengere polynomials for the funfit testing
	def p1(x):
		return x
	
	def p2(x):
		return 0.5*(3*x**2 - 1)
	
	def p3(x):
		return 0.5*(5*x**3 - 3*x)
	
	def p4(x):
		return (1/8.0)*(35*x**4 - 30*x**2 + 3)
	
	def p5(x):
		return (1/8.0)*(63*x**5 - 70*x**3 + 15*x)
	
	# Used in funfit
	fs = [p1, p2, p3, p4, p5][::-1]
	
	# Used in scipy.optimize.curve_fit
	def lagrange_5(x, a, b, c, d, e):
		return a*p1(x) + b*p2(x) + c*p3(x) + d*p4(x) + e*p5(x)
	
	# Expected results (using the data-set above)
	expected = optimize.curve_fit(lagrange_5, xs, ys)
	
	# Back to discrete
	max_error = 10**-5
	
	# Our results
	results = datafit.funfit(fs=fs, xs=xs, ys=ys, full=True)
	# Scipy uses non-linear least-squares, residue does not apply
	test_it(expected=expected, results=results, max_error=max_error, rs=False)
	
	print '	funfitc...'
	
	# Back to continuous
	max_error = 10**-3
	
	# Our results
	results = datafit.funfitc(fs=fs, f=f, xo=-1, xn=1, full=True)
	# Scipy uses non-linear least-squares, residue does not apply
	test_it(expected=expected, results=results, max_error=max_error, rs=False)
	
def t_numint():
	
	print 'numint...'
	
	def test_it(expected, result, max_error):
		
		diff = abs(expected-result)
		if diff < max_error:
			print 'OK'
		else:
			print 'FAIL'
			print '		Expected:', expected
			print '		Got:', result
			print '		Difference:', diff
			print '		Tolerance:', max_error
	
	# How much can the tests differ from the expected results? Adjust 
	# according to the method and step size used.
	max_error = 10**-6
	
	# Some functions to test
	def f1(x):
		return numpy.sin(2*numpy.pi*x)+numpy.cos(3*numpy.pi*x)
	
	def f2(x):
		return x**2 + numpy.exp(3*numpy.pi*x)
	
	def f3(x):
		return 0
	
	def f4(x):
		return x
	
	print '	Test 1/4...',
	expected = integrate.quad(f1, -1, 1)[0]
	result = numint.numint(f1, -1, 1)
	test_it(expected, result, max_error)
	
	print '	Test 2/4...',
	expected = integrate.quad(f2, -1, 1)[0]
	result = numint.numint(f2, -1, 1)
	test_it(expected, result, max_error)
	
	print '	Test 3/4...',
	expected = integrate.quad(f3, -1, 1)[0]
	result = numint.numint(f3, -1, 1)
	test_it(expected, result, max_error)
	
	print '	Test 4/4...',
	expected = integrate.quad(f4, -1, 1)[0]
	result = numint.numint(f4, -1, 1)
	test_it(expected, result, max_error)
	
if __name__ == '__main__':
	t_rootfinding()
	t_interpol()
	t_datafit()
	t_numint()
	# t_linalg()