import numpy as np
from math import pi
import pylab as pl
import networkx as nx
from time import time

try:
    from diffeqpy import de
    from julia import Main
except:
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main
    from diffeqpy import de
    # print("error")
    # exit(0)


np.random.seed(1)


julia_f = Main.eval("""
function KM_model!(du, u, p, t)
    
    @Base.Threads.threads for i = 1:p[2]
        du[i] = p[3][i] + p[1] * sum(p[4][:, i] .* sin.(u .- u[i]))
    end
    return du
end
""")

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
    p = [K/N, N, omega_0, adj]

    start = time()

    for i in range(len(K)):
        p[0] = K[i] / N
        prob = de.ODEProblem(julia_f, theta_0, tspan, p)
        sol = de.solve(prob, de.Tsit5(), saveat=dt, abstol=1e-6, reltol=1e-6);

        print("K = {:.3f}".format(K[i]))

    print("pyjulia Done in {} seconds.".format(time() - start))
    
    
    # print(type(theta))
    # print(sol.shape)
    # print(type(sol.t), len(sol.t))
    # print(type(sol.u), len(sol.u), len(sol.u[0]), type(sol.u[0]))
    

# print(type(sol.t), type(sol.u), type(sol.u[0]))