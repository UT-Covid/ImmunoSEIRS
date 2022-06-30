#!/bin/sh
#SBATCH -J ImmunoSEIRS               # Job name
#SBATCH -o saved_data/Austin_'$1'_'$RUN_NAME_ADD'_%j.out                # Name of stdout output file (%j expands to jobId)
#SBATCH -p nvdimm
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH -n 1                          # Total number of mpi tasks requested
#SBATCH -t 09:00:00                   # Run time (hh:mm:ss)
#SBATCH -A A-ib1
#SBATCH --qos=vippj_p3000

module load python
# module load Rstats # module R doesn't work well
python BA12-stoch.py

