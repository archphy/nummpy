import sympy
import numpy

def d_n(f, n):
	"""For internal use."""
	# f's nth derivative, using symbolic differentiation (sympy)
	# Returns a sympy lambda function
		
	x = sympy.symbols('x')
	try:
		d_1 = f(x)
		for i in xrange(n):
			d_1 = d_1.diff(x)
	except AttributeError:		# d/dx = 0
		d_1 = 0
	except TypeError:
		raise TypeError('f must be sympy comaptible. Supported operations are'\
		'+ * - /. For functions like sin, log... Use sympy.sin, sympy.log...'\
		'Refer to sympy manual to see what is available')
	return sympy.lambdify(x, d_1)

def tol_iter(n, reset=False):
	"""Tolerance is given by the number of iterations."""
	
	global i
	if reset:
		i = 0
		return
	else:
		try:
			i += 1
		except NameError:
			i = 0
	if i == n:
		i = 0
		return True
	else:
		return False

def max_(xo, xi):
	"""For internal use."""
	
	return xo > xi

def min_(xo, xi):
	"""For internal use."""
	
	return xo < xi

def hxy(f, xo, xn, n=0, h=0):
	"""Returns n points (x, f(x)) from xo to xn, or n' pts. with step size h."""
	
	def pts(xo, xn, h):
		try:
			xs = numpy.arange(xo,xn+h,h)
			ys = f(xs)
		except TypeError:		# Function isn't numpy-compatible
			xs, ys = [], []
			for i in numpy.arange(xo,xn+h,h):
				xs.append(i)
				ys.append(f(i))
		return (xs, ys)
	if h != 0:
		return pts(xo, xn, h)
	elif n != 0:
		return pts(xo, xn, abs(xn-xo) / (float(n)-1))
	else:
		raise ValueError('n or h must be > 0.')

def maxmin_fx(f, xs, maxmin, *args, **keyargs):
	"""For internal use."""
	# maxmin(f(x)) foreach x in xs. Returns (x, y).
	#
	# f: The function to be evaluated. (python-compatible function)
	# xs: The xs to evaluate at. (set of floats)
	# maxmin: Comparison function.* (python-compatible function)
	# *args: Any extra args f might get.
	# **keyargs: Any keyarg:value f might get.
	#
	# * Must be a two argument function returning the comparison between both.
	# It'll compare the max/min for each x in xs 'til xs[i-1] with xs[i] 
	#
	# E.g.
	# def f_(x, y):
	#  return x > y
	#
	# maxmin_fx(f, xs, f_)
	
	top_x = xs[0]
	top_y = abs(f(top_x, *args, **keyargs))
	for x in xs[1:]:
		y = abs(f(x, *args, **keyargs))
		if maxmin(y, top_y):
			top_x = x
			top_y = y
	return (top_x, top_y)