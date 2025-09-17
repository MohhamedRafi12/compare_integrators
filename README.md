# Compare different quadrature rules for integration

There are two examples provided for calculating the weights and abscissas for gaussian quadrature rules, try:

```
make
./gqconstants
```

or

```
python gqconstants.py
```

You can also use the C++ example as a guide to build your own executable

There is no need to look at rules >~25 for Gaussian quadrature.  And you can also stop at ~ 1000 divisions for the trapezoidal and Simpson's rules.  If you run much longer you'll see the numerical errors become visible for the trapezoidal, but you'll need to think about how to code efficiently or the running time may be very long.


Question Answers
  Textbook questions:
  1) See Errors.png
  2) See Errors.png
  3) See Errors.png
  4) Using a reference to N^-2, and N^-4, we can see Simpsons and Trapezoid rules following the expected behavior. For the gaussian quad, we hit machine precision much faster. For e^x, a smooth function on the interval of 0 to np.pi, we get expected behavior. For x^2 we see expected behavior but it's interesting. Since Simpsons fits to a quadratic, and gaussian quads uses polynomials, we get exact answers. 

  Class questions:
  1) No method really works well near singularities (not across the singularity)
  2) See BadErrors.png
  3) Methods that oscillate rapidly, or have singularities across the intervals break the code. I believe if you function maybe be smooth but if 
  it oscillates so quickly then it would be harder to fit a poly. to the such functions. I did cos(100x) and that breaks the convergence.
  4) You could scale your x-values that factor into your oscillators. 


