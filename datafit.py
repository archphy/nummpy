from numint import numint
import sympy
import numpy

# TODO: Multi-processing
# TODO: Option to use sympy.simplify(x, g(x))
# TODO: Use cs the other way around so we don't have to cs[::-1]

def linear(xs, ys, full=False):
	"""Linear regression for (xs, ys).
	
	xs: The x points. (numpy array of floats)
	ys: The y points. (numpy array of floats)
	full: Return just the linear regression function (False) or the function's 
	coefficients, the residue, and the function itself (True)? (bool) 
	
	E.g. linear(numpy.array([-1.0, 0.0, 1.0]), numpy.array([-1.0, 0.0, 1.0]))
	"""
	# Optimization of polyfit, since linear regression is the most common one.
	
	sxs = sum(xs)
	cs = numpy.linalg.solve(
		[[len(xs), sxs], [sxs, sum(xs**2)]],
		[sum(ys), sum(xs*ys)]
	)
	def g(x):
		return cs[0] + cs[1]*x
	if full:
		return (g, cs[::-1], rs(g, xs, ys))
	else:
		return g

def polyfit(xs, ys, n, full=False):
	"""Least squares fit for (xs, ys) using a polynomial of degree n.
	
	This is the discrete approach to the least squares problem, whereas 
	polyfitc is the continuous approach.
	
	xs: The x points. (numpy array of floats)
	ys: The y points. (numpy array of floats)
	n: The polynomial degree. (int)
	full: Return just the polynomial fit function (False) or the function's 
	coefficients, the residue, and the function itself (True)? (bool)
	
	E.g. polyfit(numpy.array([-1.0, 0.0, 1.0]), numpy.array([1.0, 0.0, 1.0]), 2) 
	"""
	
	ans = []
	coefs = []
	for i in xrange(n+1):		# Degree 2 -> c0*x^0 + c1*x^1 + c2*x^2 (n+1)
		ans.append(sum(ys*xs**i))
		coefs.append([])
		for j in xrange(i, n+1+i):
			coefs[i].append(sum(xs**j))
	cs = numpy.linalg.solve(coefs, ans)
	def g(x):
		return sum([cs[i]*x**i for i in xrange(len(cs))])
	if full:
		return (g, cs[::-1], rs(g, xs, ys))
	else:
		return g

def polyfitc(f, xo, xn, n, met='boole', h=10**-4, full=False):
	"""Least squares fit for f from xo to xn using a polynomial of degree n.
	
	This is the continuous approach to the least squares problem, whereas 
	polyfit is the discrete approach.

	f: The function to approximate. (python-compatible function)
	xo: Evaluate from xo. (float)
	xn: To xn. (float)
	n: The polynomial degree. (int)
	met: The integration method. 'sym' for symbolic integration (might take a 
	LONG time and f must be sympy-compatible), else see numint. (str or int)
	h: The step size for the numerical integration, if any. (float)
	full: Return just the polynomial fit function (False) or the function's 
	coefficients, the residue, and the function itself (True)? (bool)
	
	def f(x):
		return math.sin(x*math.pi*2) + math.cos(x*math.pi*3)
	
	E.g. polyfitc(f, -1, 1, 5, 'sym')
	"""	
	# The logic behind this can be found at
	# http://math.stackexchange.com/questions/825414/
	# A post by me showing the steps I took and asking if it's correct (it is).
	#
	# Or
	#
	# http://www3.nd.edu/~zxu2/acms40390F11/sec8-2.pdf
	# A slide from University of Notre Drame using the same algorithm.

	ans = []
	coefs = []
	x = sympy.symbols('x')	
	for i in xrange(n+1):		# Degree 2 -> c0*x^0 + c1*x^1 + c2*x^2 (n+1)
		f_ = sympy.lambdify(x, (x**i)*f(x))
		if met == 'sym':
			ans.append(sympy.integrate((x**i)*f(x), (x, xo, xn)))
		else:
			ans.append(numint(f_, xo, xn, met, h))
		coefs.append([])
		for j in xrange(i, n+1+i):
			if met == 'sym':
				coefs[i].append(sympy.integrate(x**j, (x, xo, xn)))
			else:
				coefs[i].append(numint(sympy.lambdify(x, x**j), xo, xn, met, h))
	cs = numpy.linalg.solve(coefs, ans)
	def g(x):
		return sum([cs[i]*x**i for i in xrange(len(cs))])
	if full:
		return (g, cs[::-1], rs(
					sympy.lambdify(x, (f(x)-g(x))**2), [xo, xn], None, met, h))
	else:
		return g

# TODO: Non-linear fitting
# http://en.wikipedia.org/wiki/Non-linear_least_squares
# http://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
def funfit(fs, xs, ys, full=False):
	"""Least squares fit for (xs, ys) using fs to fit the data, where fs is a 
	set of one-variable real-valued sympy-compatible functions linear in the 
	parameters*.
	
	This is the discrete approach to the least squares problem, whereas 
	funfitc is the continuous approach.
	
	* http://statpages.org/nonlin.html
	
	fs: Fit the data using this functions. (set of numpy-compatible function)
	f: The function to approximate. (sympy-compatible function)
	xo: Evaluate from xo. (float)
	xn: To xn. (float)
	full: Return just the function fit function (False) or the function's 
	coefficients, the residue, and the function itself (True)? (bool)
	
	E.g.: funfit([l0, l1, l2, l3, l4], xs, ys)
	Where ln is a function for the nth Legendre Polynomial, and (xs, ys) the 
	data-set.
	"""
	
	ans = []
	coefs = []
	n = len(fs)
	for i in xrange(n):
		try:
			ans.append(sum(ys*fs[i](xs)))
		except TypeError:		# Only catch 'cause not clear what caused it
			raise TypeError('fs functions must be numpy-compatible.')
		coefs.append([])
		for j in xrange(n):
			try:
				coefs[i].append(sum(fs[j](xs)*fs[i](xs)))
			except TypeError:		# Constant function
				coefs[i].append(sum([fs[j](xs)*fs[i](xs)]*n))
	cs = numpy.linalg.solve(coefs, ans)
	def g(x):
		return sum([cs[i]*fs[i](x) for i in xrange(len(cs))])
	if full:
		return (g, cs[::-1], rs(g, xs, ys))
	else:
		return g

def funfitc(fs, f, xo, xn, met='boole', h=10**-4, full=False):
	"""Least squares fit for f from xo to xn using fs to fit the data, where fs 
	is a set of one-variable real-valued numpy-compatible functions linear in 
	the	parameters*.
	
	This is the continuous approach to the least squares problem, whereas 
	funfit is the discrete approach.
	
	* http://statpages.org/nonlin.html
	
	fs: Fit the data using this functions. (set of numpy-compatible function)
	xs: The x points. (numpy array of floats)
	ys: The y points. (numpy array of floats)
	met: The integration method. 'sym' for symbolic integration (might take a 
	LONG time and fs must be sympy-compatible), else see numint. (str or int)
	full: Return just the function fit function (False) or the function's 
	coefficients, the residue, and the function itself (True)? (bool)
	
	E.g.: funfitc([l0, l1, l2, l3, l4], f, -1, 1)
	Where ln is a function for the nth Legendre Polynomial, and f a function.
	"""
	
	ans = []
	coefs = []
	x = sympy.symbols('x')
	n = len(fs)
	for i in xrange(n):
		if met == 'sym':
			ans.append(sympy.integrate(f(x)*fs[i](x), (x, xo, xn)))
		else:
			ans.append(numint(sympy.lambdify(x, f(x)*fs[i](x)), xo, xn, met, h))
		coefs.append([])
		for j in xrange(n):
			if met == 'sym':
				coefs[i].append(sympy.integrate(fs[j](x)*fs[i](x), (x, xo, xn)))
			else:
				coefs[i].append(
					numint(sympy.lambdify(x, fs[j](x)*fs[i](x)), xo, xn, met, h)
				)				
	cs = numpy.linalg.solve(coefs, ans)
	def g(x):
		return sum([cs[i]*fs[i](x) for i in xrange(len(cs))])
	if full:
		return (g, cs[::-1], rs(
					sympy.lambdify(x, (f(x)-g(x))**2), [xo, xn], None, met, h))
	else:
		return g

# TODO: option to use symbolic integration
def rs(g=None, xs=None, ys=None, met='boole', h=10**-4):
	"""Returns the residue of the data fitting function.
	
	Discrete case:
	
		The residue is given by the sum of (ys-g(xs))^2 foreach x in xs.
		
		g: The data-fitting function. (python-compatible function).
		xs: The x points used in the fitting. (set of floats)
		ys: The y points used in the fitting. (set of floats)
		met: Do not supply. (None)
		h: Do not supply. (None)
			
	Continuous case:
			
		The residue is given by the integral of (f(x)-g(x))^2dx from a to b.
	
		g: (f(x)-g(x))^2, where f is the original function and g the data-
		-fitting one. (python-compatible function)
		xs: The initial and final points of the integration interval, i.e. 
		a e b*. (set of floats).
		ys: Do not supply. (None)
		met: The numerical integration method. (str or int) (see numint)
		h: The step size for the numerical integration. (float)
		* There can be values between a and b, those will be ignored.
			
	E.g. 
	
	Discrete
		rs(g, [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0], None, None)
	Continuous
		rs(f_minus_g_square, [1.0, 10.0], None, 'boole', 10**-4)	
	"""
	
	if ys is not None:
		return sum([(ys[i]-g(xs[i]))**2 for i in xrange(len(xs))])
	elif g:
		return numint(g, xs[0], xs[-1], met, h)
	else:
		raise ValueError('Not enough information. Check help(rs).')	