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
function f!(du,u,p,t)
  du[1] = p[1] * u[1] - p[2] * u[1]*u[2]
  du[2] = -3 * u[2] + u[1]*u[2]
end
""" )
julia_g = Main.eval("""
function g!(du,u,p,t)
  du[1] = p[3]*u[1]
  du[2] = p[4]*u[2]
end
""")

prob_func = Main.eval("""
function prob_func(prob, i, repeat)
    remake(prob, u0 = rand(2))
    end
""")

# def prob_func(prob, i, repeat):
#     de.remake(prob, u0=np.random.rand(2))


if __name__ == "__main__":

    tspan = (0.0, 10.0)
    p = [1.5, 1.0, 0.1, 0.1]
    prob = de.SDEProblem(julia_f, julia_g, [1.0, 1.0], tspan, p)
    ensemble_prob = de.EnsembleProblem(prob, prob_func=prob_func)
    
    sol = de.solve(ensemble_prob,
                   de.SRIW1(),
                   trajectories=2)
    
    # print(type(sol))
    print(len(sol.u))
    print(len(sol.u[0]))
    print(len(sol.u[0][0]))

