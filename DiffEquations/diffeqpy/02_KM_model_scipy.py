import numba
import numpy as np
from math import pi
import pylab as pl
import networkx as nx
from time import time
from scipy.integrate import odeint

np.random.seed(1)


def KM_model(theta, t, p):

    K, N, omega, adj = p
    dtheta = np.zeros(len(theta))
    n = len(theta)

    for i in range(n):
        dtheta[i] = omega[i] + K/N * \
            np.sum(adj[i, :] * np.sin(theta - theta[i]))

    return dtheta


def order(theta):
    # calculate the order parameter

    r = np.zeros(len(theta))
    for i in range(len(theta)):
        r[i] = abs(sum(np.exp(1j * theta[i]))) / N
    return r
#-------------------------------------------------------------------#


K = np.arange(0.0, 1.05, 0.05)    # coupling
N = 50                          # number of nodes
dt = 0.01                       # time step
T = 500.0                       # simulation time
T_trans = 0.0                   # transition time
mu = 0.0                        # mean value of initial frequencies
sigma = 0.15                    # std of initial frequencies
p = 0.5                         # probability of connections in random graph


G = nx.gnp_random_graph(N, p, seed=1)
adj = nx.to_numpy_array(G, dtype=int)


if __name__ == "__main__":

    ind_transition = int(T_trans/dt)
    theta_0 = np.random.uniform(-pi, pi, size=N)
    omega_0 = np.random.normal(mu, sigma, size=N)

    tspan = np.arange(0.0, T, 0.01)
    p = [K, N, omega_0, adj]

    # fig, ax = pl.subplots(1, figsize=(5, 4))

    start = time()

    R = np.zeros(len(K))
    for k in range(len(K)):
        p[0] = K[k]
        theta = odeint(KM_model, theta_0, tspan, args=(p,))
        # theta = theta[ind_transition:, :]   # drop transition time
        # r = order(theta)
        # R[k] = np.average(r)

        print("K = {:.3f}, R = {:.3f}".format(K[k], R[k]))

    print("Done in {} seconds".format(time() - start))
    
    # ax.plot(K, R, marker="o", lw=1, c="k")
    # ax.set_xlabel("K")
    # ax.set_ylabel("R")
    # np.savez("data/scipy_R", R=R, K=K)
    # pl.savefig("data/scipy_R.png")
    # pl.close()


# print(type(sol.t), type(sol.u), type(sol.u[0]))