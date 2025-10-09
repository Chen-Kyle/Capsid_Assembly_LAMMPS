#!/bin/bash
#SBATCH --job-name=test_parallel_pentamer
#SBATCH --partition=skx
#SBATCH --nodes=2
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=all
##SBATCH -mail-type=end
##SBATCH -mail-type=fail
#SBATCH --mail-user=smritipradhan@brandeis.edu
source /home1/09816/smritipradhan/.bashrc
module load pylauncher/4.7
conda init
conda activate westpa_openmm
#source env.sh
# Create a temporary job file
export WEST_SIM_ROOT="/work2/09816/smritipradhan/stampede3/dimer_to_pentamer/unbiased"
export SIM_NAME=$(basename $WEST_SIM_ROOT)

rm -f commandlines
Enative=(0.5 0.75 1.0)
total_nodes=$SLURM_JOB_NUM_NODES
total_Enative=${#Enative[@]}

total_combinations=$total_Enative
nruns=$((total_nodes*48/total_combinations))
echo $nruns

for enative in "${Enative[@]}";do
for i in $(seq 1 $nruns); do
	mkdir -p $enative/$i
	echo "python run_oligomer_simulation_preferbinding_iter.py  --Erep 0.15 --Enative $enative --run $i > bound_capsid__${enative}_${i}.txt" >> commandlines
done
done
python test_launch.py


