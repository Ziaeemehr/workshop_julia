import numpy as np
import pylab as pl
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
function f!(du, u, p, t)
    du .= 1.01 * u
end
""")

prob_func = Main.eval("""
function prob_func(prob, i, repeat)
    remake(prob, u0 = rand() * prob.u0)
    end
""")

# def prob_func(prob, i, repeat):
#     de.remake(prob, u0=rand() * prob.u0)


if __name__ == "__main__":

    tspan = (0.0, 10.0)
    prob = de.ODEProblem(julia_f, [0.5], tspan)
    ensemble_prob = de.EnsembleProblem(
        prob,
        prob_func=prob_func)
    
    sol = de.solve(ensemble_prob,
                   de.Tsit5(),
                   de.EnsembleThreads(),
                   trajectories=2,
                   saveat=0.1)
    # print(type(sol))
    print(len(sol.u))
    print(len(sol.u[0]))
    print(len(sol.u[0][0]))

