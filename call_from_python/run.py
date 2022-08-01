from julia.api import Julia
import numpy as np
import julia

jl = Julia(compiled_modules=False)
j = julia.Julia()


fn = j.include("test.jl")

x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
X = fn(x)
print(X)