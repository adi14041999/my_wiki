# Independence of Random Variables

**Independence** is one of the most fundamental and important concepts in probability theory. It allows us to simplify complex calculations, understand the structure of random phenomena, and make powerful assumptions that lead to elegant mathematical results.

Two random variables $X$ and $Y$ are **independent** if and only if their joint probability distribution factors into the product of their individual distributions.

**For discrete random variables:**

$$P(X = x, Y = y) = P(X = x) \cdot P(Y = y) \quad \text{for all } x, y$$

**For continuous random variables:**

$$f_{X,Y}(x, y) = f_X(x) \cdot f_Y(y) \quad \text{for all } x, y$$

**For n random variables:**

A collection of random variables $X_1, X_2, \ldots, X_n$ is **mutually independent** if and only if their joint CDF factors into the product of their individual CDFs:

$$F_{X_1, X_2, \ldots, X_n}(x_1, x_2, \ldots, x_n) = F_{X_1}(x_1) \cdot F_{X_2}(x_2) \cdots F_{X_n}(x_n) = \prod_{i=1}^n F_{X_i}(x_i) \quad \text{for all } x_1, x_2, \ldots, x_n$$

**Equivalently, for continuous random variables, the joint PDF factors:**

$$f_{X_1, X_2, \ldots, X_n}(x_1, x_2, \ldots, x_n) = f_{X_1}(x_1) \cdot f_{X_2}(x_2) \cdots f_{X_n}(x_n) = \prod_{i=1}^n f_{X_i}(x_i) \quad \text{for all } x_1, x_2, \ldots, x_n$$

**And for discrete random variables, the joint PMF factors:**

$$P(X_1 = x_1, X_2 = x_2, \ldots, X_n = x_n) = P(X_1 = x_1) \cdot P(X_2 = x_2) \cdots P(X_n = x_n) = \prod_{i=1}^n P(X_i = x_i) \quad \text{for all } x_1, x_2, \ldots, x_n$$

Independence means that **knowing the value of one random variable gives you no information about the value of the other**. The random variables are completely unrelated in their behavior.