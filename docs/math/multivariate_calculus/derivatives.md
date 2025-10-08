# Derivatives

## Scalar case

You are probably familiar with the concept of a derivative in the scalar case:
given a function f : R → R, the derivative of f at a point x ∈ R is defined as:

f'(x) = lim_{h→0} (f(x + h) - f(x))/h

Derivatives are a way to measure change. In the scalar case, the derivative
of the function f at the point x tells us how much the function f changes as the
input x changes by a small amount ε:

f(x + ε) ≈ f(x) + εf'(x)

For ease of notation we will commonly assign a name to the output of f,
say y = f(x), and write ∂y/∂x for the derivative of y with respect to x. This
notation emphasizes that ∂y/∂x is the rate of change between the variables x and
y; concretely if x were to change by ε then y will change by approximately ε ∂y/∂x.

We can write this relationship as

x → x + Δx ⟹ y → y + (∂y/∂x)Δx

You should read this as saying "changing x to x + Δx implies that y will
change to approximately y + Δx ∂y/∂x". This notation is nonstandard, but I like
it since it emphasizes the relationship between changes in x and changes in y.