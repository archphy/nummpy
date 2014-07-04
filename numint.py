import numpy

def trapezoidal(xo, xn, f):
	"""For internal use."""
	
	return (xn - xo) * (f(xo) + f(xn))/2.0

def simpson(xo, xn, f):
	"""For internal use."""
	
	return ((xn - xo)/6.0) * (f(xo) + 4*f((xo+xn)/2.0) + f(xn))

def boole(xo, xn, f):
	"""For internal use."""
	
	h = (xn - xo) / 4.0
	return (2*h/45.0)*(7*f(xo)+32*f(xo+h)+12*f(xo+2*h)+32*f(xo+3*h)+7*f(xn))

def numint(f, xo, xn, met='boole', h=10**-4):
	"""Integrate f numerically from xo to xn with optional step size h.
	
	xo: The initial point of the integration interval. (float)
	xn: The final point of the integration interval. (float)
	f: The function to integrate. (python-compatible function)
	g: The desired method. Default is Boole's Rule.	
	If met == 1 or met == 'trapezoidal', the Trapezoidal Rule shall be used.
	If met <= 3 or met == 'simpson', Simpson's Rule shall be used.
	Se met <= 5 or met == 'boole', Boole's Rule shall be used.
	(str or int)
	
	Note: Only simple integration of one-variable functions is supported.
	For double or tripal integration of/or multi-variable functions, use the 
	function that does symbolic instead of numerical integration, if available.
	
	E.g.
	
	def f(x):
		return math.sin(2*math.pi*x) + math.cos(3*math.pi*x)
		 
	numint(f, -1, 1)
	"""
	
	if h != 0:
		ans = 0
		for i in numpy.arange(xo,xn,h):
			ans += numint(f, i, i+h, met, h=0)
		return ans	
	if met == 1 or met == 'trapezoidal':
		return trapezoidal(xo, xn, f)
	elif met <= 3 or met == 'simpson':
		return simpson(xo, xn, f)
	elif met <= 5 or met=='boole':
		return boole(xo, xn, f)
	else:
		raise TypeError('Invalid method. Check help(numint).')
	
	
	