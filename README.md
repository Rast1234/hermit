hermit
======

Hermite Polynomial Interpolation

See `main()` for usage. Everything is hard-coded, no interaction or user-defined files, so, go to code!

Basically, you will need:

* `function` and its derivatives (up to `[highest multiplicity in nodes]-1`)
* `list of nodes` for interpolation

Example of node:

`Node(("one", 42), 3)` # named *"one"*, has value *42* and multiplicity *3*.

Function and its derivatives are stored inside `def f(x, dx=0):`, where `f(x,0)` is ***f(x)***, `f(x,1)` is ***f'(x)*** and so on.


Results are printed into `output.tex` and compiled into pdf. Graphics saved in `figure.png`. If you want to enable interactive graphics powered by `matplotlib`, uncomment `plt.show()` line inside `def plot()`.
