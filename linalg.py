import numpy

# TODO: Review and test all
# If I remember correctly, only the iterative methods are stable (but VERY slow)

# def decomp(m, cond, els):
# 	"""For internal use. Use the L_jacobi, D_jacobi, U_jacobi wrappers."""
# 		
# 	n = len(m)
# 	ans = []
# 	for i in xrange(n):
# 		ans.append([])
# 		for j in xrange(n):
# 			if eval(cond):
# 				ans[i].append(m[i][j])
# 			else:
# 				ans[i].append(els)
# 	return numpy.asarray(ans)
# 
# def L_jacobi(m):
# 	"""Returns the matrix in the L form, from the LDU representation of the 
# 	Jacobi iterative method for solving linear systems of equations.
# 	"""
# 	
# 	return decomp(m, 'i > j', 0)
# 
# def D_jacobi(m):
# 	"""Returns the matrix in the D form, from the LDU representation of the 
# 	Jacobi iterative method for solving linear systems of equations.
# 	"""
# 	
# 	return decomp(m, 'i == j', 0)
# 
# def U_jacobi(m):
# 	"""Returns the matrix in the U form, from the LDU representation of the 
# 	Jacobi iterative method for solving linear systems of equations.
# 	"""
# 	
# 	return decomp(m, 'i < j', 0)
# 
# def pivotize(m):
# 	"""Partialy pivotize m."""
# 	
# 	from copy import copy
# 	n = len(m)
# 	p = numpy.eye(n).tolist()
# 	m_ = copy(m)
# 	for i in xrange(n):
# 		col = []
# 		for j in xrange(n):
# 			col.append(abs(m_[j][i]))
# 		k = col.index(max(col))		# Line of the column's maximum
# 		if i != k and abs(m_[k][k]) < abs(m_[k][i]) and m[i][k] != 0:
# 			p[i], p[k] = p[k], p[i]
# 			m_ = numpy.array(p).dot(m_)
# 	p = numpy.asarray(p)
# 	return p, p.dot(m)
# 
# def LUP(m):
# 	"""LU permutation. Returns L, U, P, where P is the permutation matrix."""
# 	
# 	n = len(m)
# 	P, U = pivotize(m)
# 	L = decomp(U, 'i > j', 1)
# 	L = decomp(L, 'i >= j', 0)
# 	for i in xrange(n):
# 		for j in xrange(i):  
# 			L[i][j] = float(U[i][j])/U[j][j]
# 			U[i] -= U[j]*L[i][j]
# 	return L, U, P
# 
# def solve(a, b, LUP_=None):
# 	"""Solve the system of linear equations ax=b using LU decomposition.
# 	
# 	If LUP_ (set of 3 matrices) is supplied, use it instead of decomposing a.
# 	"""
# 	
# 	if LUP_ == None:
# 		L, U, P = LUP(a)
# 	else:
# 		L, U, P = LUP_
# 	try:
# 		return (numpy.linalg.inv(U).dot(numpy.linalg.inv(L))).dot(P.dot(b))
# 	except numpy.linalg.linalg.LinAlgError:		# Singular matrix
# 		raise Exception('The system has a number of solutions != 1.')
# 	
# def solveLUP(L, U, P, b):
# 	"""Solves the LU decomposed system with solution b."""
# 	
# 	return solve(None, b, (L, U, P))
# 
# def diagdom(m):
# 	"""Is the matrix diagonally dominant?"""
# 	
# 	n = len(m)
# 	sum_ = 0
# 	for i in xrange(n):
# 		for j in xrange(n):
# 			if i != j:
# 				sum_ += abs(m[i][j])
# 		if abs(m[i][i]) <= sum_:
# 			return False
# 	return True
# 
# def conv(a, met):
# 	"""Tests if the iterative method met converges for the a matrix.
# 	
# 	met can be either 'jacobi' or 'gauss-seidel'. There is no need for calling 
# 	this function before using said methods, since it is called by them anyway.
# 	"""
# 	
# 	if not diagdom(a):
# 		try:
# 			print 'Not diagonal dominant, taking eigenvalues (migth take'\
# 			'longer than actually solving the system, Ctrl+C to stop)'
# 			if met == 'gauss-seidel':
# 				B = numpy.linalg.inv(L_jacobi(a)+D_jacobi(a)).dot(U_jacobi(a))
# 			elif met == 'jacobi':
# 				B = numpy.linalg.inv(D_jacobi(a)).dot(L_jacobi(a)+U_jacobi(a))
# 			if abs(max(numpy.linalg.eig(B)[0])) >= 1:
# 				return False
# 		except numpy.linalg.linalg.LinAlgError:		# Singular matrix
# 			raise numpy.linalg.linalg.LinAlgError('Impossible to verify if the'\
# 											' method converges for the matrix.')
# 	return True
# 
# def dov1(a, b, v0, dd, n, met):
# 	"""For internal use."""
# 	
# 	v1 = []
# 	sum_ = 0
# 	for i in xrange(n):
# 		for j in xrange(n):
# 			if i != j:
# 				if met == 'gauss-seidel':
# 					try:
# 						sum_ += a[i][j]*v1[j]
# 					except IndexError:
# 						sum_ += a[i][j]*v0[j]
# 				elif met == 'jacobi':
# 					sum_ += a[i][j]*v0[j]
# 		v1.append((b[i] - sum_) / dd[i])
# 		sum_ = 0
# 	return numpy.array(v1)
# 
# def solveiter(a, b, v0, e, met='gauss-seidel', conv_=False):
# 	"""Solves the system of linear equations ax = b iteratively.
# 	
# 	a: The matrix of coefficients. (numpy 2D array of floats)
# 	b: The results matrix. (numpy 1D array of floats)
# 	v0: The initial guess for the solution. (numpy 1D array of floats)
# 	e: Stop when max(abs(current_estimative-initial_guess)) < e. (float)
# 	met: The method to be used - 'jacobi' or 'gauss-seidel'. (str)
# 	conv_: Check if the method converges? (bool)
# 	
# 	E.g. solveiter(numpy.array([[4]]), numpy.array([0]), [0.1], 10**-16)
# 	"""
# 
# 	global iter
# 	iter = 0
# 	if conv_ or conv(a, met):
# 		n = len(a)
# 		dd = a.diagonal()
# 		v1 = dov1(a, b, v0, dd, n, met)
# 		while max(abs(v1-v0)) >= e:
# 			iter += 1
# 			v0 = v1
# 			v1 = dov1(a, b, v0, dd, n, met)
# 		iter = 0
# 		return v1
# 	else:
# 		raise Exception('Method diverges for the matrix.')