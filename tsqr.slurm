#!/bin/bash
#SBATCH -J tsqr
#SBATCH -o tsqr.out
#SBATCH -e tsqr.err
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00

cd "$SLURM_SUBMIT_DIR" || exit 1

module purge
module load gcc/13.2.0-gcc-8.5.0-sfeapnb
module load mpi/2021.12

make clean
make
make run
