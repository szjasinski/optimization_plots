# Szymon Jasinski

import numpy as np
import matplotlib.pyplot as plt


# distributions - uniform, normal, cauchy

# chromosome - binary string of length d


# initialization - random initial population
def rand_population_binary(m, n):
    # m chromosomes of size n
    return [np.random.randint(2, size=n) for _ in range(m)]


def rand_population_uniform(m, a, b):
    # m design points
    x1 = np.random.uniform(low=a, high=b, size=m)
    x2 = np.random.uniform(low=a, high=b, size=m)
    return np.dstack((x1, x2))


def rand_population_normal(m, mean, cov):
    # m design points
    x1 = np.random.multivariate_normal(mean, cov, size=m)
    x2 = np.random.multivariate_normal(mean, cov, size=m)
    return np.dstack((x1, x2))


def rand_population_cauchy(m):
    # m design points
    x1 = np.random.standard_cauchy(size=m)
    x2 = np.random.standard_cauchy(size=m)
    return np.dstack((x1, x2))



# selection - producing a list of m parental pairs for m children of the next generation


def truncation_selection(k, y):
    # truncation selection - sample parents from the best k chromosomes in the population
    pass


def tournament_selection(k, pop, f):
    # tournament selection - each parent is the fittest out of k randomly chosen chromosomes in the population
    rng = np.random.default_rng()
    pop = pop.reshape(-1,2)

    def get_parent(k, pop, f):
        pop_sample = rng.choice(pop, k)
        y = np.apply_along_axis(f, axis=1, arr=pop_sample)
        idx = np.argmin(y)
        parent = pop_sample[idx]
        return parent

    return np.array([(get_parent(k, pop, f), get_parent(k, pop, f)) for _ in range(pop.shape[0])])


def roulette_wheel_selection(k, y):
    # roulette wheel selection (fitness proportionate selection) - each parent is chosen with a probability
    # proportional to its performance relative to the population
    pass


# crossover - combines the chromosomes of parents to form children


def single_point_crossover(a, b):
    p = np.random.randint(0, len(a)+1)
    return np.hstack((a[:p], b[p:]))


def uniform_crossover(a, b):
    pass


# mutation

def gaussian_mutation(sigma, child):
    # adding zero-mean Gaussian noise
    return child + np.random.normal(0, sigma, size=len(child))


def plot_algorithm(results):

    for k, pop in enumerate(results['populations']):
        ax.set_title(f'k = {k}')

        gray_shade = str((k + 1) / len(results['populations']))
        ax.scatter(pop[:, 0], pop[:, 1], linewidth=5 - 5 * k / len(results['populations']), c=gray_shade)
        plt.pause(0.4)  # has to be at the end to update plot

    plt.show()


def genetic_algorithm(f, population, k_max, S, C, M, sigma, selection_k):
    """

    :param f: objective function
    :param population: initial population
    :param k_max: number of iterations
    :param S: selection method
    :param C: crossover method
    :param M: mutation method
    :param sigma: standard deviation for gaussian mutation method
    :param selection_k: k for selection method
    :return:
    """

    results = {"populations": []}

    for k in range(k_max):
        parents = S(selection_k, population, f)
        children = [C(parents[i, 0], parents[i, 1]) for i in range(parents.shape[0])]
        population = np.array([M(sigma, child) for child in children])
        results["populations"].append(population)

        # print("iteration k =", k)
        # print("parents")
        # print(parents)
        # print("children")
        # print(children)
        # print("population")
        # print(population)

    minimum = np.apply_along_axis(f, axis=1, arr=population.reshape(-1,2)).min()
    print(minimum)

    plot_algorithm(results)


# def f(x):
#     return x[0]**2 + x[1]**2
#
#
# def F(x, y):
#     return x**2 + y**2


# def f(x):
#     return x[0]**2 - x[1]**2
#
#
# def F(x, y):
#     return x**2 - y**2


def f(x):
    return np.sin(x[0]) + np.sin(x[1])


def F(x, y):
    return np.sin(x) + np.sin(y)


# create 2d grid
delta = 0.025  # grid step
a = 15  # half of square length
x_arr = np.arange(-a, a, delta)
y_arr = np.arange(-a, a, delta)
X, Y = np.meshgrid(x_arr, y_arr)

# visualize colored level set of defined function
fig, ax = plt.subplots(figsize=(9.5, 8))
Z = F(X, Y)  # Z is a function of X,Y from np.meshgrid - easy to plot
CS = ax.contourf(X, Y, Z)  # draw level set
fig.colorbar(CS)


# pop = rand_population_uniform(100, -10, 10)

pop = rand_population_normal(100, [0, 0], [[15, 0], [0, 15]])

# pop = rand_population_cauchy(100)

genetic_algorithm(f=f, population=pop, k_max=20, S=tournament_selection,
                  C=single_point_crossover, M=gaussian_mutation, sigma=0.4, selection_k=3)

