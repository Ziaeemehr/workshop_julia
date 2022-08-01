import DifferentialEquations
import Plots

pl = Plots
df = DifferentialEquations

f(u,p,t) = 0.98u
u0 = 1.0
tspan = (0.0, 1.0)
prob = df.ODEProblem(f, u0, tspan)
sol = df.solve(prob)

# pl.display(pl.plot(sol))

