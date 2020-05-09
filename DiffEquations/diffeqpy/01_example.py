import matplotlib.pyplot as plt
from diffeqpy import de

def f(u,p,t):
    return -u

u0 = 0.5
tspan = (0., 1.)
prob = de.ODEProblem(f, u0, tspan)
sol = de.solve(prob)

plt.plot(sol.t,sol.u)
plt.show()