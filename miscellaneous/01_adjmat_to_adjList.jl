using PyCall


function adjmat_to_adjlist(A, threshold::Float64=0.5)
    adjlist=Array{Int64}[]
    row, col = size(A)
    for i =1:row 
        tmp = []
        for j =1:col 
            if (abs(A[i,j] > threshold))
                push!(tmp, j)
            end
        end
        push!(adjlist, tmp)
    end
    adjlist
end

nx = pyimport("networkx")
np = pyimport("numpy")

G = nx.gnp_random_graph(5, 0.6, seed=1)
adj = nx.to_numpy_array(G, dtype=np.int)
row, col = size(adj)

# println(typeof(adj[1,1]))

adjlist = adjmat_to_adjlist(adj)
# println(typeof(adjlist), size(adjlist))
# adjlist
