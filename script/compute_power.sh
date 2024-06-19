#!/bin/bash
#SBATCH --nodes 3
#SBATCH --partition HENON
#SBATCH --ntasks-per-node 100
#SBATCH --time 24:00:00
#SBATCH --job-name hscy3
#SBATCH -o log/hscy3-map.out
#SBATCH -e log/hscy3-map.error

source /hildafs/projects/phy200017p/share/ana/setupLsstim1.sh
mpirun -np 300 $SCRATCH/HSC_Y3_reanalysis/script/compute_power.py --nside 4096 --mpi

