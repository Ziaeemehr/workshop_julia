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
    a = p[4]
    if kind ==2 
        @Base.Threads.threads for i = 1:p[2]
            du[i] = p[3][i] + p[1] * sum(a[:, i] .* sin.(u .- u[i]))
        end
    elseif kind==1
        @Base.Threads.threads for i = 1:p[2]
            du[i] = p[3][i] + p[1] * 0.5 * sum(a[:, i] .* (1.0 .- cos.(u .- u[i])))
        end
    else
        println("unknown scillator type")
        exit()
    end
    return du
end
""")

# p = [K[0] / N, N, omega_0, adj[0], kind, noise_amp]

julia_g = Main.eval("""
function sKM_model!(du, u, p, t)
    @Base.Threads.threads for i = 1:p[2]
        du[i] = p[6]
    end
    du
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


K = np.arange(0.0, 0.5, 0.02)    # coupling
N = 10                          # number of nodes
dt = 0.01                       # time step
t_final = 500.0                       # simulation time
t_trans = 10.0                   # transition time
mu = 2.0                        # mean value of initial frequencies
sigma = 0.0                     # std of initial frequencies
# p = 0.5                         # probability of connections in random graph
kind = [1, 2]
noise_amp = 0.001

G = nx.gnp_random_graph(N, 0.5, seed=1)
ER = nx.to_numpy_array(G, dtype=int)

# sf_dag = np.loadtxt("SF-DAG.txt", dtype=int).T
# adj = [sf_dag]  #, adj_FB, adj_FF]
adj = [ER]
net_labels = ["ER"]  # ["SF-DAG"]

if __name__ == "__main__":

    # ind_transition = int(t_trans/dt)
    theta_0 = np.random.uniform(-pi, pi, size=N)
    omega_0 = np.random.normal(mu, sigma, size=N)

    tspan = (0.0, t_final)
    p = [K[0] / N, N, omega_0, adj[0], kind, noise_amp]
    R = {}

    fig, ax = pl.subplots(1, figsize=(5, 4))

    start = time()

    for n in range(len(adj)):
        p[3] = adj[n]
        for k in range(len(kind)):
            p[4] = kind[k]
            r = []
            for i in range(len(K)):
                p[0] = K[i] / N
                prob = de.SDEProblem(julia_f, julia_g, theta_0, tspan, p)
                # sol = de.solve(prob, de.EM(),
                #             # saveat=0.1,
                #             # saveat=np.arange(t_trans, t_final, dt),
                #             dt=dt,
                #             abstol=1e-6,
                #             reltol=1e-6,
                #             )
                sol = de.solve(prob, de.SOSRA(),
                               saveat=0.1,
                               dt=dt,
                               )
                r.append(np.mean(order(sol.u, step=1)))
                print("adj = {:s}, kind = {:d}, K = {:.3f}, R = {:.3f}".format(
                    net_labels[n], kind[k], K[i], r[-1]))
            R[net_labels[n] + str(k + 1)] = r

    print("Done in {} seconds.".format(time() - start))

    for n in range(len(adj)):
        for k in range(len(kind)):
            ax.plot(K, R[net_labels[n] + str(k + 1)],
                    label=net_labels[n] + str(k + 1))

    ax.legend()
    ax.set_xlabel("K")
    ax.set_ylabel(r"$\langle R \rangle$")
    pl.savefig("sde_mat.png")
    pl.close()
    # pl.show()
