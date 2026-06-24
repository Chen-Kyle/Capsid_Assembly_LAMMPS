for i in $(seq 1 60);
do 
	cp cg_AB_avg_connectivity.txt cg_A${i}B${i}_connectivity.txt
	cp cg_CD_connectivity.txt cg_C${i}D${i}_connectivity.txt
done
