import numba
import numpy as np
from math import pi
import pylab as pl
import networkx as nx
from time import time
from diffeqpy import de
from scipy.integrate import odeint

np.random.seed(1)


def KM_model(dtheta, theta, p, t):

    for i in range(p[1]):
        dtheta[i] = p[2][i] + p[0] * \
            np.sum(p[3][i, :] * np.sin(theta - theta[i]))

    return dtheta


def order(theta):
    # calculate the order parameter

    r = np.zeros(len(theta))
    for i in range(len(theta)):
        r[i] = abs(sum(np.exp(1j * theta[i]))) / N
    return r
#-------------------------------------------------------------------#


K = np.arange(0.0, 1.05, 0.05)    # coupling
N = 100                          # number of nodes
dt = 0.05                       # time step
T = 200.0                       # simulation time
T_trans = 0.0                   # transition time
mu = 2.0                        # mean value of initial frequencies
sigma = 0.1                    # std of initial frequencies
p = 0.5                         # probability of connections in random graph


G = nx.gnp_random_graph(N, p, seed=1)
adj = nx.to_numpy_array(G, dtype=int)


if __name__ == "__main__":

    ind_transition = int(T_trans/dt)
    theta_0 = np.random.uniform(-pi, pi, size=N)
    omega_0 = np.random.normal(mu, sigma, size=N)

    tspan = (0.0, T)
    p = [K[0]/N, N, omega_0, adj]
    numba_KM = numba.jit(KM_model)
    # numba_order = numba.jit(order)

    # fig, ax = pl.subplots(1, figsize=(5, 4))

    start = time()

    R = np.zeros(len(K))
    for k in range(len(K)):
        p[0] = K[k] / N
        prob = de.ODEProblem(numba_KM, theta_0, tspan, p)
        sol = de.solve(prob, de.Tsit5(), saveat=dt, abstol=1e-6, reltol=1e-6);
        # theta = np.asarray(sol.u)

        # theta = theta[ind_transition:, :]   # drop transition time
        # r = numba_order(theta)
        # R[k] = np.average(r)

        print("K = {:.3f}".format(K[k]))

    print("numba Done in {} seconds".format(time() - start))
    
    # ax.plot(K, R, marker="o", lw=1, c="k")
    # ax.set_xlabel("K")
    # ax.set_ylabel("R")
    # np.savez("data/numba_R", R=R, K=K)
    # pl.savefig("data/numba_R.png")
    # pl.close()


# print(type(sol.t), type(sol.u), type(sol.u[0]))