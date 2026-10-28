<!-- qsub -l walltime=04:00:00  -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=test_distance_distrib,N_CHUNKS=50,OFFSET=0 submit.pbs -->

qsub -J 1-3 -l walltime=00:30:00 -l select=1:ncpus=8:mem=64gb -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=test_distance_distrib,N_CHUNKS=50 submit.pbs

below is qsub commands history - managing manually right now to avoid extra waits / extra computational overhead

## QSUB commands history - 27 oct 2026

first ran for -J 1-3, have a quick look at logs to see everything's working as expected, adjust walltimes and run for -J 4-25 

have about 400-600 synthetic datasets in total, using 25 chunks at the moment, per chunk we have 10-30 datasets

run the pooled version to catch datasets per batch that go out of mem (useful for manifold stats mostly, due to count matrices interacting with isomap) -- dont recommend running this usually as major errors won't be raised

### 1000,10

#### distance_distrib_stats - done

50 chunks test:
```qsub -J 1-3 -l walltime=00:30:00 -l select=1:ncpus=8:mem=64gb -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=test_distance_distrib,N_CHUNKS=50 run_pyspoc.pbs```

25 chunks test on 1-3 chunks:
```qsub -J 1-3 -l walltime=00:30:00 -l select=1:ncpus=1:mem=1gb -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=distance_distrib_stats,N_CHUNKS=25 run_pyspoc.pbs```

25 chunks full run on 4-25 chunks:
```qsub -J 4-25 -l walltime=00:20:00 -l select=1:ncpus=2:mem=1gb -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=distance_distrib_stats,N_CHUNKS=25 run_pyspoc.pbs```

#### pyspoc internal config - done

25 chunks test on 1-3 chunks: ```qsub -J 1-3 -l walltime=00:30:00 -l select=1:ncpus=1:mem=1gb -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/pyspoc_internal_stats.yaml,CONFIG_NAME=pyspoc_internal_stats,N_CHUNKS=25 run_pyspoc.pbs```

25 chunks run on 4-25 chunks: ```qsub -J 4-25 -l walltime=00:30:00 -l select=1:ncpus=2:mem=1gb -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/pyspoc_internal_stats.yaml,CONFIG_NAME=pyspoc_internal_stats,N_CHUNKS=25 run_pyspoc.pbs```

#### manifold stats - in progress

25 chunks test on 1-3 chunks: ```qsub -N mf_1000_10 -J 1-3 -l walltime=01:00:00 -l select=1:ncpus=8:mem=16gb -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/manifold_stats.yml,CONFIG_NAME=manifold_stats,N_CHUNKS=25 run_pyspoc.pbs```

POOLED 25 chunks run on 1-25 chunks: ```qsub -N mf_1000_10 -J 1-25 -l walltime=00:30:00 -l select=1:ncpus=8:mem=16gb -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/manifold_stats.yml,CONFIG_NAME=manifold_stats,N_CHUNKS=25 run_pyspoc_pooled.pbs```

### 1000,100

#### distance_distrib_stats - done

25 chunks test on 1-3 chunks: ```qsub -J 1-3 -l walltime=00:30:00 -l select=1:ncpus=2:mem=1gb -v MANIFEST=metadata_files/syn_data_N1000_P100.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=distance_distrib_stats,N_CHUNKS=25 run_pyspoc.pbs```

25 chunks run on 4-25 chunks: ```qsub -J 4-25 -l walltime=00:20:00 -l select=1:ncpus=2:mem=1gb -v MANIFEST=metadata_files/syn_data_N1000_P100.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=distance_distrib_stats,N_CHUNKS=25 run_pyspoc.pbs```

#### pyspoc internal config - done

25 chunks test on 1-3 chunks: ```qsub -N pic_test_1000_100 -J 1-3 -l walltime=01:00:00 -l select=1:ncpus=6:mem=6gb -v MANIFEST=metadata_files/syn_data_N1000_P100.csv,CONFIG=configs/pyspoc_internal_stats.yaml,CONFIG_NAME=pyspoc_internal_stats,N_CHUNKS=25 run_pyspoc.pbs```
(didnt work at 1gb mem)

25 chunks run on 4-25 chunks: ```qsub -N pic_test_1000_100 -J 4-25 -l walltime=01:00:00 -l select=1:ncpus=6:mem=6gb -v MANIFEST=metadata_files/syn_data_N1000_P100.csv,CONFIG=configs/pyspoc_internal_stats.yaml,CONFIG_NAME=pyspoc_internal_stats,N_CHUNKS=25 run_pyspoc.pbs```

#### manifold stats - todo

### 1000,1000

#### distance_distrib_stats - done

25 chunks test on 1-3 chunks: ```qsub -N dist_1000_1000 -J 1-3 -l walltime=00:30:00 -l select=1:ncpus=2:mem=2gb -v MANIFEST=metadata_files/syn_data_N1000_P1000.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=distance_distrib_stats,N_CHUNKS=25 run_pyspoc.pbs```

25 chunks on 4-25 chunks: ```qsub -N dist_1000_1000 -J 4-25 -l walltime=00:30:00 -l select=1:ncpus=2:mem=2gb -v MANIFEST=metadata_files/syn_data_N1000_P1000.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=distance_distrib_stats,N_CHUNKS=25 run_pyspoc.pbs```

#### pyspoc internal config - in progress

25 chunks test on 1-3 chunks: ```qsub -N pic_test_1000_1000 -J 1-3 -l walltime=10:00:00 -l select=1:ncpus=16:mem=32gb -v MANIFEST=metadata_files/syn_data_N1000_P1000.csv,CONFIG=configs/pyspoc_internal_stats.yaml,CONFIG_NAME=pyspoc_internal_stats,N_CHUNKS=25 run_pyspoc.pbs```
(running 936)

#### manifold stats - todo

## other helpful commands

qstat -x -f <jobid> | grep resources_used

grep -L 'DONE!' pbs_job_outs/<jobid>*.OU | wc -l