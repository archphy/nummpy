nummpy (NUMerical Methods Python library) supplies a narrow range of numerical methods 
and auxiliary functions. It has a syntax similar to numpy's one. It is not 
intended to replace existing libraries - numpy and scipy are faster, have more 
features and are more robust. This is a personal project maintained as a hobby. 
It comes with ABSOLUTELY NO WARRANTY.

For help, use help(method).

For copying, see LICENSE.

The example image shows the function sin(2*pi*x)+cos(3*pi*x) from -1 to 1, 
(green line), 20 evenly spaced points, the Lagrange Interpolation Polynomial of 
degree 19 (blue line, overlapped by the green one), and the data fitting (via 
linear least squares) of the 20 pts using the six first Legendre Polynomials.
   
![Screenshot](https://raw.github.com/a442/nummpy/master/ex.png "Screenshot")

The code to do it:

```python
>>> from nummpy import *
>>> import numpy
>>> import sympy
>>> from matplotlib import pyplot
>>> def fn(x):          
...     return numpy.sin(2*numpy.pi*x)+numpy.cos(3*numpy.pi*x)
... 
>>> def p0(x):
...     return 1
... 
>>> def p1(x):
...     return x
... 
>>> def p2(x):
...     return 0.5*(3*x**2 - 1)
... 
>>> def p3(x):
...     return 0.5*(5*x**3 - 3*x)
... 
>>> def p4(x):
...     return (1/8.0)*(35*x**4 - 30*x**2 + 3)
... 
>>> def p5(x):
...     return (1/8.0)*(63*x**5 - 70*x**3 + 15*x)
... 
>>> fs = [p0, p1, p2, p3, p4, p5]
>>> xs, ys = aux.hxy(fn, -1, 1, n=20)
>>> vxs = numpy.arange(-1, 1, 10**-5)
>>> x = sympy.symbols('x')
>>> interpol_g = sympy.lambdify(x, interpol.lagrange(x, xs, ys))
>>> datafit_g = datafit.funfit(fs, xs, ys)
>>> pyplot.plot(xs, ys, 'gs')
[<matplotlib.lines.Line2D object at 0x4b25c90>]
>>> pyplot.plot(vxs, interpol_g(vxs), 'b')
[<matplotlib.lines.Line2D object at 0x3a5aad0>]
>>> pyplot.plot(vxs, datafit_g(vxs), 'r')
[<matplotlib.lines.Line2D object at 0x4b2a610>]
>>> pyplot.plot(vxs, fn(vxs), 'g')
[<matplotlib.lines.Line2D object at 0x4b2ab10>]
>>> pyplot.show()
```