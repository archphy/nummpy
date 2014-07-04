from aux import d_n

def bisec(f, a, b, ft, **keyargs):
	"""Returns an approximation of the function's root via the bisection method.
	
	Assumes that the function is appropriate for this method, i.e., in the 
	[a, b] interval where the root is being searched, the function
	- Is continuous
	- Has one and only one root
	
	f: The function. (python-compatible function)
	a: The interval's initial point. (float)
	b: The interval's end point. (float)
	ft: The tolerance function. (python-compatible function)
	**keyargs: Any keyword:value the tolerance function might take. (dict)
	
	E.g. bisec(f, 2.0, 4.0, aux.tol_iter, n=10**3)
	Note: It might be necessary to modify this function for some tolerance 
	functions. As is, it checks if ft(keyargs) is True in order to stop.
	"""
	
	m = (a+b)/2.0
	if ft(**keyargs):		# Tolerance reached, approximation
		return m
	if f(a)*f(m) < 0: 
		return bisec(f, a, m, ft, **keyargs)
	elif f(b)*f(m) < 0:
		return bisec(f, m, b, ft, **keyargs)
	elif f(m) == 0:		# Exact root
		return m
	else:
		raise ValueError("f(a)*f(b) > 0, discontinuous or no-root interval.")
		# For the interval to be continuous, the functions' limit must exist at 
		# the critical points of the interval. In order to exist only one root 
		# at the interval, f'(x) must preserve it's sign at [a, b].

def newton(f, xo, ft, **keyargs):
	"""Returns an approximation of the function's root via newton's method.
	
	Assumes that the function is appropriate for this method, i.e., in the
	neighborhood of xo, the function
	- Is continuous
	- Has one and only one root
	- Has it's first derivative != 0
	
	f: The function. (python-compatible function)
	xo: The initial guess. (float)
	ft: The tolerance function. (python-compatible function)
	**keyargs: Any keyword:value the tolerance function might take. (dict)
	
	E.g. newton(fx, 1.5, aux.tol_iter, n=10**1)
	Note: It might be necessary to modify this function for some tolerance 
	functions. As is, it checks if ft(keyargs) is True in order to stop.
	"""
	
	if ft(**keyargs):
		return xo
	else:
		try:
			return newton(f, xo - (f(xo) / d_n(f, 1)(xo)), ft, **keyargs)
		except ZeroDivisionError:
			raise ValueError('d/dx f must be != 0')

def secant(f, xoo, xo, ft, **keyargs):
	"""Returns an approximation of the function's root via the secant method.
	
	Assumes that the function is appropriate for this method, i.e.
	- It is continuous on the nighborhood of xoo and of xo
	- Has one and only one root in said neighborhood
	- xo != xoo
	
	f: The function. (python-compatible function)
	xoo: The initial guess xn-1. (float)		
	xo: The initial guess xn. (float)
	ft: The tolerance function. (python-compatible function)
	**keyargs: Any keyword:value the tolerance function might take. (dict)

	E.g. secant(fx, 1.0, 1.5, aux.tol_iter, n=10**1)
	Note: It might be necessary to modify this function for some tolerance 
	functions. As is, it checks if ft(keyargs) is True in order to stop.
	"""
	
	if ft(**keyargs):
		return xo
	else:
		try:
			return secant(
				f,
				xo,
				xo - f(xo) * ((xo - xoo) / (f(xo) - f(xoo))),
				ft,
				**keyargs
			)
		except ZeroDivisionError:
			raise ValueError('xoo must be != xo')