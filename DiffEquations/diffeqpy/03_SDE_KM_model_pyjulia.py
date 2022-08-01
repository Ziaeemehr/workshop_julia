import numpy as np
from math import pi
import pylab as pl
import networkx as nx
from time import time
from diffeqpy import de
from julia import Main

np.random.seed(1)


julia_KM = Main.eval("""
function KM_model!(du, u, p, t)
    
    @Base.Threads.threads for i = 1:p[2]
        du[i] = p[3][i] + p[1] * sum(p[4][:, i] .* sin.(u .- u[i]))
    end
    du
end
""" )

julia_sKM = Main.eval("""
function sKM_model!(du, u, p, t)
    @Base.Threads.threads for i = 1:p[2]
        du[i] = p[5]
    end
    du
end
""")


def order(theta):
    # calculate the order parameter
    step = 10
    a = range(0, len(theta), step)
    r = np.zeros(len(a))
    for i in range(len(a)):
        j = a[i]
        r[i] = abs(sum(np.exp(1j * theta[j]))) / N
    return r
#-------------------------------------------------------------------#


K = np.arange(0.0, 1.0, 0.05)  # coupling
# K = np.array([0.5])
N = 100                         # number of nodes
dt = 0.01                       # time step
T = 200.0                       # simulation time
T_trans = 0.0                   # transition time
mu = 2.0                        # mean value of initial frequencies
sigma = 0.1                     # std of initial frequencies
p = 0.5                         # probability of connections in random graph
noise_amp = 0.1

G = nx.gnp_random_graph(N, p, seed=1)
adj = nx.to_numpy_array(G, dtype=int)


if __name__ == "__main__":

    ind_transition = int(T_trans/dt)
    theta_0 = np.random.uniform(-pi, pi, size=N)
    omega_0 = np.random.normal(mu, sigma, size=N)

    tspan = (0.0, T)
    p = [K/N, N, omega_0, adj, noise_amp]

    start = time()

    R = np.zeros(len(K))
    for i in range(len(K)):
        p[0] = K[i] / N
        prob = de.SDEProblem(julia_KM, julia_sKM, theta_0, tspan, p)
        # sol = de.solve(prob, de.EM(), dt=dt)
        sol = de.solve(prob, de.SOSRA(), dt=dt, saveat=0.1)
        R[i] = np.mean(order(sol.u))
        # saveat=dt, abstol=1e-6, reltol=1e-6);

        print("K = {:.3f}, R = {:.6f}".format(K[i], R[i]))

    print("pyjulia Done in {} seconds.".format(time() - start))

    pl.plot(K, R, marker="o")
    pl.xlabel("K")
    pl.ylabel("R")
    pl.tight_layout()
    pl.savefig("SDE_KM_SOSRA.png")
    pl.show()
