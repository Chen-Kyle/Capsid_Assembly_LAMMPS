#!/bin/bash
#module load share_modules/NAMD/2.13b2_mpi_sp
# Set WESTPA-related variables

export WEST_SIM_ROOT="$PWD"
export SIM_NAME=$(basename $WEST_SIM_ROOT)

# Set runtime commands
#export NAMD=/share/software/scisoft/NAMD/2.13b2/mpi_sp_Linux-x86_64-icc.arch/namd2
#for i in {1..1000}
#do
#random_number=$RANDOM
#random_number1=$(echo $random_number + 1 | bc) 
#echo $random_number
#echo $random_number1
#done
