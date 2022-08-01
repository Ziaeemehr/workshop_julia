- multithreading

```sh
julia -t $num_threads 
```

- slurm example:

```sh
#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=...
#SBATCH --mem-per-cpu=400M
#SBATCH --time=00:10:00

module load StdEnv/2020 julia/1.5.2
julia -t $SLURM_CPUS_PER_TASK heavyThreads.jl
```


- multiprocessing

```sh
julia -p $num_process
```

- slurm example

```
#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --cpu-per-task=1
#SBATCH --mem-per-cpu=400M
#SBATCH --time=00:10:00
srun hostname -s > hostfile # parallel I/O
sleep 5
julia --machine-file
```
