import numpy as np
from math import pi
import pylab as pl
from sys import exit

try:
    from diffeqpy import de
    from julia import Main
except:
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main
    from diffeqpy import de


julia_f = Main.eval("""
function f!(du, u, p, t)
    for i = 1:p[1]
        du[i] = 2.0 + sum(sin.(u .- u[i]))
    end
    return du
end
""")

julia_g = Main.eval("""
function g!(du, u, p, t)
    for i = 1:p[1]
        du[i] = p[2]
    end
    return du
end
""")

prob_f = Main.eval("""
function prob_func(prob, i, repeat)
    DifferentialEquations.remake(prob, u0 = rand(prob.p[1]) .* 2.0 .* pi .- pi)
    end
""")

#-------------------------------------------------------------------#


K = 2.0                        # coupling
N = 2                         # number of nodes
dt = 0.01                      # time step
t_final = 5.0                 # simulation time
t_trans = 1.0                 # transition time
mu = 2.0                       # mean value of initial frequencies
sigma = 0.0                    # std of initial frequencies
kind = 2
noise_amp = 0.005

if __name__ == "__main__":

    np.random.seed(1)
    theta_0 = np.random.uniform(-pi, pi, size=N)
    tspan = (0.0, t_final)

    p = [N, noise_amp]
    prob = de.SDEProblem(julia_f, julia_g, theta_0, tspan, p)
    # sol = de.solve(prob, de.SOSRA())
    ensemble_prob = de.EnsembleProblem(prob, prob_func=prob_f)
    sim = de.solve(ensemble_prob,
                   de.SRIW1(),
                   de.EnsembleThreads(),
                   trajectories=3,
                   saveat=0.1)

    print(len(sim.u))
    print(len(sim.u[0]))
    print(len(sim.u[0][0]))
    print(type(sim.u))