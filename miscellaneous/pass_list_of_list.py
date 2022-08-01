import matplotlib.pyplot as plt
import sys
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

    # println(length(p))
    println(p[2][1])
    n = length(u)
    for i = 1:n
        du[i] = - sin(u[i])
    end
    du
end
""" )

julia_sf = Main.eval("""
function sf!(du, u, p, t)
    n = length(u)
    for i = 1:n
        du[i] = 0.001
    end
    return du
end
""" )

# def f(u, p, t):
#     return - u
    



u0 = [0.5, 1.0]
p = [10, [[1, 2, 3], [4, 5]]]
# print(type(p[1]))
# sys.exit(0)
tspan = (0., 2)
# prob = de.ODEProblem(f, u0, tspan, p)
prob = de.SDEProblem(julia_f, julia_sf, u0, tspan, p)
sol = de.solve(prob, de.SOSRA(), dt=0.01, saveat=0.1, seed=1)

# prob = de.SDEProblem(julia_f, julia_sf, u0, tspan, p)
# sol = de.solve(prob, de.SOSRA(), dt=dt, saveat=0.1, seed=1)
# sol = de.solve(prob, de.EM(), dt=0.01)
plt.plot(sol.t,sol.u)
plt.show()