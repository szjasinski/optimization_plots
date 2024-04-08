from sympy import diff
from sympy import Symbol, lambdify
from sympy import sin, cos, exp, oo, tan, acos, log, sqrt

import matplotlib.pyplot as plt

import numpy as np


x1 = Symbol('x1')

# DEFINE FUNCTION -- examples
# f = - 2 * x1**5
# f = x1**2
# f = x1**3
# f = - x1**3 + 2*x1**2 - 2
# f = - x1**5 + 2*x1**2 - 2
# f = sin(x1*2)
f = 32*x1**6 - 48*x1**4 + 18*x1**2 - 1  # a=1.1, x = -0.3

# DEFINE INITIAL POINT X
x = -0.3


delta = 0.025  # grid step
a = 1.1  # half of square length
x_arr = np.arange(-a, a, delta)


fig, ax = plt.subplots(figsize=(9.5, 8))
F = lambdify([x1], f)  # lambdify - transform SymPy expressions to lambda functions
Z = F(x_arr)  # Z is a function of X,Y from np.meshgrid - easy to plot
ax.set_facecolor("#1CC4AF")
ax.plot(x_arr, Z)


def get_grid_point(x_input):
    x_index = np.argmin((x_arr - x_input)**2)
    return x_arr[x_index]


def visualize_newton_method(f, results):

    for k, (x, delta) in enumerate(zip(results['x'], results['delta'])):
        print('k:', k, 'delta:', delta, '\nx:', np.array(x).T, '\nf(x):', f.subs({x1: x}), '\n')
        ax.set_title(f'k = {k}, delta = {delta}\nx1 = {x}\nf(x) = {f.subs({x1: x})}')

        p = get_grid_point(x)

        # draw lines
        if k >= 1:
            recent_x = results['x'][k-1]
            ax.plot([recent_x, x], [f.subs({x1: recent_x}), f.subs({x1: x})], 'r')

        # draw points
        gray_shade = str((k + 1) / len(results['x']))
        ax.scatter(p, f.subs({x1: p}), linewidth=5 - 5 * k / len(results['x']), c=gray_shade, zorder=2)

        plt.pause(0.3)  # has to be at the end to update plot

    plt.show()


def secant_method(f, x, epsilon, k_max):

    k, delta = 1, epsilon + 1
    results = {'x': [x], 'delta': [delta]}

    recent_x = x * 1.01

    while abs(delta) > epsilon and k <= k_max:

        first_derivative = diff(f, x1).subs({x1: x}).evalf()

        # approximate second derivative
        second_derivative = (x - recent_x) / (diff(f, x1).subs({x1: x}).evalf() - diff(f, x1).subs({x1: recent_x}).evalf())

        delta = first_derivative * second_derivative

        recent_x = x
        x -= delta
        k += 1

        results['x'].append(x)
        results['delta'].append(delta)

    visualize_newton_method(f, results)


# initiate newton method
secant_method(f, x, epsilon=0.01, k_max=30)



