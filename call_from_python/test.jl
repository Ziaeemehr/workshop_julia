function myArrayFn(x::Array{Float64})
    println("array size: $(size(x))");
    println("max element: $(maximum(x))")
    println("min element: $(minimum(x))")
    return 2x
end