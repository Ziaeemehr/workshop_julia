import numpy as np
from math import pi
import pylab as pl
import networkx as nx
from time import time
from sys import exit
from time import time

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
function f!(du, u, p, t)
    
    kind = p[5]
    adjlist = p[4]
    if kind == 2 
        @Base.Threads.threads for i = 1:p[2]
            du[i] = p[3][i] + p[1] * sum(sin.(u[adjlist[i]] .- u[i]))
        end
    elseif kind == 1
        @Base.Threads.threads for i = 1:p[2]
            du[i] = p[3][i] + p[1] * 0.5 * sum(1.0 .- cos.(u[adjlist[i]] .- u[i]))
        end
    else
        println("unknown scillator type")
        exit()
    end
    return du
end
""")
# p = [K[0] / N, N, omega_0, adjlist, kind, noise_amp]

julia_g = Main.eval("""
function g!(du, u, p, t)
    @Base.Threads.threads for i = 1:p[2]
        du[i] = p[6]
    end
    du
end
""")

prob_func = Main.eval("""
function prob_func(prob, i, repeat)
    DifferentialEquations.remake(prob, u0 = rand(prob.p[2]) .* 2.0 .* pi .- pi)
end
""")


def order(theta, step=5):
    # calculate the order parameter
    N = len(theta)  # 10
    nstep = len(theta[0])  # 501
    a = np.arange(0, nstep, step)
    r = np.zeros(len(a))
    for i in range(len(a)):
        x = np.array([theta[j][a[i]] for j in range(N)])
        r[i] = abs(sum(np.exp(1j * x))) / N
    return r


def adjmat_to_adjlist(A, threshold=0.5):

    adjlist = []
    row, col = A.shape
    for i in range(row):
        tmp = []
        for j in range(col):
            if (abs(A[i, j] > threshold)):
                tmp.append(j+1)  # +1 for difference in indexing
        adjlist.append(tmp)

    return adjlist

#-------------------------------------------------------------------#


K = np.arange(0.0, 0.8, 0.05)
N = 200
dt = 0.01
t_final = 500.0
t_trans = 10.0
mu = 2.0
sigma = 0.0
# p = 0.5
kind = [1, 2]
noise_amp = 0.001
num_sim = 2

# G = nx.gnp_random_graph(N, 0.5, seed=1)
# ER = nx.to_numpy_array(G, dtype=int)

sf_dag = np.loadtxt("SF-DAG.txt", dtype=int)
adj = [sf_dag]  # , adj_FB, adj_FF]
adjlist = [adjmat_to_adjlist(sf_dag)]
net_labels = ["SF-DAG"]  # ["SF-DAG"]

if __name__ == "__main__":

    # ind_transition = int(t_trans/dt)

    theta_0 = np.random.uniform(-pi, pi, size=N)
    omega_0 = np.random.normal(mu, sigma, size=N)

    tspan = (0.0, t_final)
    p = [K[0] / N, N, omega_0, adjlist, kind, noise_amp]
    R = {}

    fig, ax = pl.subplots(1, figsize=(5, 4))

    start = time()
    times = np.arange(t_trans, t_final, dt)

    for n in range(len(adjlist)):
        p[3] = adjlist[n]
        for k in range(len(kind)):
            p[4] = kind[k]
            rk = []
            for i in range(len(K)):

                p[0] = K[i] / N
                prob = de.SDEProblem(julia_f, julia_g, theta_0, tspan, p)
                ensemble_prob = de.EnsembleProblem(prob,
                                                   prob_func=prob_func)
                sim = de.solve(ensemble_prob,
                               de.SOSRA(),
                               de.EnsembleThreads(),
                               trajectories=num_sim,
                               saveat=0.1)

                rsi = np.empty(num_sim)
                for si in range(num_sim):
                    rsi[si] = np.mean(order(sim.u[si], step=1))

                rk.append(np.mean(rsi))

                print("adj = {:s}, kind = {:d}, K = {:.3f}, R = {:.3f}".format(
                    net_labels[n], kind[k], K[i], rk[-1]))

            R[net_labels[n] + str(k + 1)] = rk

    print("Done in {} seconds.".format(time() - start))

    for n in range(len(adj)):
        for k in range(len(kind)):
            ax.plot(K, R[net_labels[n] + str(k + 1)],
                    label=net_labels[n] + str(k + 1))

    ax.legend()
    ax.set_xlabel("K")
    ax.set_ylabel(r"$\langle R \rangle$")
    pl.savefig("sde_adjlist_ensemble.png")
    pl.close()


# print(len(sim.u),
#       len(sim.u[0]),
#       len(sim.u[0][0]),
#       )
# print(type(sim.u[0]), len(sim.u[0]), "\n",
#       type(sim.u[0][0]), len(sim.u[0][0]), "\n",
#       type(sim.u[0][0][0]))
