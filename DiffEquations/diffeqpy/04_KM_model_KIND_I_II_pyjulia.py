import numpy as np
from math import pi
import pylab as pl
import networkx as nx
from time import time
from sys import exit

try:
    from diffeqpy import de
    from julia import Main
except:
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main
    from diffeqpy import de


np.random.seed(1)


julia_f = Main.eval("""
function KM_model!(du, u, p, t)
    
    kind = p[5]
    if kind ==2 
        for i = 1:p[2]
            du[i] = p[3][i] + p[1] * sum(p[4][:, i] .* sin.(u .- u[i]))
        end
    elseif kind==1
        for i = 1:p[2]
            du[i] = p[3][i] + p[1] * 0.5 * sum(p[4][:, i] .* (1.0 .- cos.(u .- u[i])))
        end
    else
        println("unknown scillator type")
        exit()
    end
    return du
end
""")


def order(theta, step=5):
    # calculate the order parameter
    N = len(theta[0])
    a = np.arange(0, len(theta), step)
    r = np.zeros(len(a))
    for i in range(len(a)):
        r[i] = abs(sum(np.exp(1j * theta[a[i]]))) / N
    return r
#-------------------------------------------------------------------#


K = np.arange(0.0, 0.3, 0.01)    # coupling
N = 3                          # number of nodes
dt = 0.05                       # time step
t_final = 500.0                       # simulation time
t_trans = 10.0                   # transition time
mu = 2.0                        # mean value of initial frequencies
sigma = 0.0                     # std of initial frequencies
# p = 0.5                         # probability of connections in random graph
kind = [1, 2]

# G = nx.gnp_random_graph(N, p, seed=1)
# adj = nx.to_numpy_array(G, dtype=int)

adj_FB = np.array([[0, 0, 1],
                   [1, 0, 0],
                   [0, 1, 0]], dtype=int).T

adj_FF = np.array([[0, 0, 0],
                   [1, 0, 0],
                   [1, 1, 0]], dtype=int).T

adj = [adj_FB, adj_FF]
net_label = ["FB", "FF"]

if __name__ == "__main__":

    # ind_transition = int(t_trans/dt)
    theta_0 = np.random.uniform(-pi, pi, size=N)
    omega_0 = np.random.normal(mu, sigma, size=N)

    tspan = (0.0, t_final)
    p = [0, N, omega_0, 0, kind]
    R = {}

    fig, ax = pl.subplots(1, figsize=(5, 4))

    start = time()

    for n in range(2):
        p[3] = adj[n]
        for k in range(2):
            p[4] = kind[k]
            r = []
            for i in range(len(K)):

                print("adj = {:s}, kind = {:d}, K = {:.3f}".format(
                    net_label[n], kind[k], K[i]))

                p[0] = K[i] / N
                prob = de.ODEProblem(julia_f, theta_0, tspan, p)
                sol = de.solve(prob, de.Tsit5(),
                            # saveat=0.1,
                            # saveat=np.arange(t_trans, t_final, dt),
                            dt=dt,
                            abstol=1e-9,
                            reltol=1e-9)
                r.append(np.mean(order(sol.u, step=1)))
            R[net_label[n] + str(k + 1)] = r

    print("pyjulia Done in {} seconds.".format(time() - start))

    for n in range(2):
        for k in range(2):
            ax.plot(K, R[net_label[n] + str(k + 1)],
                    label=net_label[n] + str(k + 1))

    ax.legend()
    ax.set_xlabel("K")
    ax.set_ylabel(r"$\langle R \rangle$")
    pl.savefig("data/loop.png")
    # pl.show()
