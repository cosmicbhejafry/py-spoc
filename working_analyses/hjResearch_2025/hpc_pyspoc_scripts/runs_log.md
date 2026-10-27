<!-- qsub -l walltime=04:00:00  -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=test_distance_distrib,N_CHUNKS=50,OFFSET=0 submit.pbs -->

qsub -J 1-3 -l walltime=00:30:00 -l select=1:ncpus=8:mem=64gb -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=test_distance_distrib,N_CHUNKS=50 submit.pbs

## QSUB commands history

qsub -J 1-3 -l walltime=00:30:00 -l select=1:ncpus=8:mem=64gb -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=test_distance_distrib,N_CHUNKS=50 run_pyspoc.pbs

qsub -J 1-3 -l walltime=00:30:00 -l select=1:ncpus=1:mem=1gb -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=distance_distrib_stats,N_CHUNKS=25 run_pyspoc.pbs

<!-- qsub -J 1-25 -l walltime=02:00:00 -l select=1:ncpus=1:mem=2gb -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=distance_distrib_stats,N_CHUNKS=25 run_pyspoc.pbs -->

## other helpful commands

qstat -x -f <jobid> | grep resources_used