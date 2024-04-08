from sympy import ordered, Matrix, hessian
from sympy import Symbol, lambdify
from sympy import sin, cos, exp, oo, tan, acos, log, sqrt

import matplotlib.pyplot as plt

import numpy as np


x1, x2 = Symbol('x1'), Symbol('x2')  # define symbols for variables

# DEFINE FUNCTION -- examples
# f = x1**2 + x2**2
# f = - x1**2 - x2**2
# f = - x1**2 + x2**2  # converges to saddle point
f = 0.5*x1**2 + 0.2*x2**6
# f = sin(x1) + sin(x2)   # points: (3,3), (-3,-3), (6,-6)

# f = (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2  # booth function
# f = x1**2 + x2  # error - matrix not invertible
# f = sin(x1 * x2)  # converges to point x such that f(x) = 1 or f(x) = -1


# f = 1/exp(sin(log(x1)*x2) + cos(x1*log(x2)))  # (1.5, 1.5), (4.5, 6), (4.8, 6.8)

# f = exp(x1*x2)

# DEFINE INITIAL POINT X
x = Matrix([6, 6])


# create 2d grid
delta = 0.025  # grid step
a = 8  # half of square length
x_arr = np.arange(-a, a, delta)
y_arr = np.arange(-a, a, delta)
X, Y = np.meshgrid(x_arr, y_arr)

# visualize colored level set of defined function
fig, ax = plt.subplots(figsize=(9.5, 8))
F = lambdify([x1, x2], f)  # lambdify - transform SymPy expressions to lambda functions
Z = F(X, Y)  # Z is a function of X,Y from np.meshgrid - easy to plot
CS = ax.contourf(X, Y, Z)  # draw level set
fig.colorbar(CS)


def get_grid_point(x_input, y_input):
    x_index = np.argmin((x_arr - x_input) ** 2)
    y_index = np.argmin((y_arr - y_input) ** 2)
    return X[0][x_index], Y[y_index][0]


def get_gradient(f, x):
    v = list(ordered(f.free_symbols))
    grad = Matrix([f]).jacobian(v)
    return grad.subs({x1: x[0], x2: x[1]}).T.evalf()


def get_hessian(f, x):
    v = list(ordered(f.free_symbols))
    return hessian(f, v).subs({x1: x[0], x2: x[1]}).evalf()


def visualize_newton_method(f, results):

    for k, (x, delta) in enumerate(zip(results['x'], results['delta.norm'])):
        print('k:', k, 'delta:', delta, '\nx:', np.array(x).T, '\nf(x):', f.subs({x1: x[0], x2: x[1]}), '\n')
        ax.set_title(f'k = {k}, delta.norm = {delta}\nx1 = {x[0]}\nx2 = {x[1]}\nf(x) = {f.subs({x1: x[0], x2: x[1]})}')

        p1, p2 = get_grid_point(x[0], x[1])
        gray_shade = str((k+1)/len(results['x']))
        ax.scatter(p1, p2, linewidth=5 - 5 * k / len(results['x']), c=gray_shade)
        plt.pause(0.3)  # has to be at the end to update plot

    plt.show()


def newton_method(f, x, epsilon, k_max):

    k, delta = 1, Matrix([oo, oo])
    results = {'x': [x], 'delta.norm': [delta.norm()]}

    while delta.norm() > epsilon and k <= k_max:

        gradient = get_gradient(f, x)
        hessian = get_hessian(f, x)
        delta = hessian.inv() * gradient

        x -= delta
        k += 1

        results['x'].append(x)
        results['delta.norm'].append(delta.norm())

    visualize_newton_method(f, results)


# initiate newton method
newton_method(f, x, epsilon=0.00000000001, k_max=50)

