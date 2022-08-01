using DifferentialEquations
using PyCall
import Random
# using Printf

Random.seed!(1234)

function KM_model!(du, u, p, t)

    n = length(u)
    for i = 1:n
        du[i] = p * sum(sin.(u .- u[i]))
    end

    du
end



K = [0.5, 0.6] #[0.0:0.05:1.05]             # coupling
N = 50                          # number of nodes
dt = 0.01                       # time step
T = 20.0                       # simulation time
T_trans = 0.0                   # transition time
mu = 0.0                        # mean value of initial frequencies
sigma = 0.15                    # std of initial frequencies
p = 0.5                         # probability of connections in random graph


# adj = read("adj.txt")
nx = pyimport("networkx")
np = pyimport("numpy")
pl = pyimport("matplotlib.pyplot")

G = nx.gnp_random_graph(N, p, seed=1)
adj = nx.to_numpy_array(G, dtype=np.int)
println("size of adj is ", size(adj, 1), ", ", size(adj, 2))
# @printf "size of adj is %d, %d\n" size(adj, 1) size(adj, 2)


ind_transition = Int64(T_trans/dt)
theta_0 = rand(N) .* 2.0 .* pi .- pi
omega_0 = randn(N) .* sigma .+ mu 
tspan = (0.0, T)
p = K[1]/N


# fig, ax = pl.subplots(1, figsize=(5, 4))
# start = time()

# R = np.zeros(length(K))
prob = ODEProblem(KM_model!, theta_0, tspan, p)
sol = solve(prob, saveat=0.01)
println(length(sol.t), length(sol.u), length(sol.u(1)))
#     # theta = np.asarray(sol.u)

#     # theta = theta[ind_transition:, :]   # drop transition time
#     # r = numba_order(theta)
#     # R[k] = np.average(r)

#     print("K = {:.3f}, R = {:.3f}".format(K[k], R[k]))
# print("Done in {} seconds".format(time() - start))

# ax.plot(K, R, marker="o", lw=1, c="k")
# ax.set_xlabel("K")
# ax.set_ylabel("R")
# np.savez("data/numba_R", R=R, K=K)
# pl.savefig("data/numba_R.png")
# pl.close()


# pl.show()
# print(type(sol.t), type(sol.u), type(sol.u[0]))