### Bash on Linux/OSX:
export JULIA_NUM_THREADS=4

### C shell on Linux/OSX, CMD on Windows:
set JULIA_NUM_THREADS=4


julia> Threads.nthreads()
julia> Threads.threadid()

julia>  Threads.@threads for i = 1:10
            a[i] = Threads.threadid()
        end