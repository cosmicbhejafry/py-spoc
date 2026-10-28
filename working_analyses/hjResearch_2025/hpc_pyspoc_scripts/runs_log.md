<!-- qsub -l walltime=04:00:00  -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=test_distance_distrib,N_CHUNKS=50,OFFSET=0 submit.pbs -->

qsub -J 1-3 -l walltime=00:30:00 -l select=1:ncpus=8:mem=64gb -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=test_distance_distrib,N_CHUNKS=50 submit.pbs

below is qsub commands history - managing manually right now to avoid extra waits / extra computational overhead

todo's:
<!-- [] generate counts cpm folders, update metadata files -->
[] rerun all with counts cpm + log-transformed
[] run all on pyspoc_internal_stats_fast.yaml

## QSUB commands history

first ran for -J 1-3, have a quick look at logs to see everything's working as expected, adjust walltimes and run for -J 4-25 

have about 400-600 synthetic datasets in total, using 25 chunks at the moment, per chunk we have 10-30 datasets

run the pooled version to catch datasets per batch that go out of mem (useful for manifold stats mostly, due to count matrices interacting with isomap) -- dont recommend running this usually as major errors won't be raised

### gtex runs

```qsub -J 1-25 -l walltime=00:20:00 -l select=1:ncpus=2:mem=1gb -v MANIFEST=metadata_files/gtex_N100_P100.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=distance_distrib_stats,N_CHUNKS=25 run_pyspoc.pbs```

```qsub -J 1-25 -l walltime=00:20:00 -l select=1:ncpus=4:mem=4gb -v MANIFEST=metadata_files/gtex_N100_P100.csv,CONFIG=configs/manifold_stats.yml,CONFIG_NAME=manifold_stats,N_CHUNKS=25 run_pyspoc.pbs```

```qsub -J 1-25 -l walltime=00:30:00 -l select=1:ncpus=4:mem=8gb -v MANIFEST=metadata_files/gtex_N100_P100.csv,CONFIG=configs/pyspoc_internal_stats_fast.yaml,CONFIG_NAME=pyspoc_internal_stats_fast,N_CHUNKS=25 run_pyspoc.pbs```


### 1000,10 (done)

#### distance_distrib_stats - done

25 chunks full run on 1-25 chunks:
```qsub -J 1-25 -l walltime=00:20:00 -l select=1:ncpus=2:mem=1gb -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=distance_distrib_stats,N_CHUNKS=25 run_pyspoc.pbs```

#### pyspoc_internal_stats_fast - done

25 chunks run: ```qsub -J 1-25 -l walltime=01:00:00 -l select=1:ncpus=2:mem=1gb -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/pyspoc_internal_stats_fast.yaml,CONFIG_NAME=pyspoc_internal_stats_fast,N_CHUNKS=25 run_pyspoc.pbs```

#### manifold stats - done (most, some out of memory)

25 chunks test on 1-3 chunks: ```qsub -N mf_1000_10 -J 1-3 -l walltime=01:00:00 -l select=1:ncpus=8:mem=16gb -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/manifold_stats.yml,CONFIG_NAME=manifold_stats,N_CHUNKS=25 run_pyspoc.pbs```

POOLED 25 chunks run on 1-25 chunks: ```qsub -N mf_1000_10 -J 1-25 -l walltime=00:30:00 -l select=1:ncpus=8:mem=16gb -v MANIFEST=metadata_files/syn_data_N1000_P10.csv,CONFIG=configs/manifold_stats.yml,CONFIG_NAME=manifold_stats,N_CHUNKS=25 run_pyspoc_pooled.pbs```

### 1000,100 (done)

#### distance_distrib_stats - done

25 chunks test on 1-3 chunks: ```qsub -J 1-3 -l walltime=00:30:00 -l select=1:ncpus=2:mem=1gb -v MANIFEST=metadata_files/syn_data_N1000_P100.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=distance_distrib_stats,N_CHUNKS=25 run_pyspoc.pbs```

25 chunks run on 4-25 chunks: ```qsub -J 1-25 -l walltime=00:45:00 -l select=1:ncpus=2:mem=1gb -v MANIFEST=metadata_files/syn_data_N1000_P100.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=distance_distrib_stats,N_CHUNKS=25 run_pyspoc.pbs```

#### pyspoc_internal_stats_fast - done

(didnt work at 1gb mem)
40 chunks run pooled: ```qsub -N pic_test_1000_100 -J 1-40 -l walltime=02:00:00 -l select=1:ncpus=6:mem=6gb -v MANIFEST=metadata_files/syn_data_N1000_P100.csv,CONFIG=configs/pyspoc_internal_stats_fast.yaml,CONFIG_NAME=pyspoc_internal_stats_fast,N_CHUNKS=40 run_pyspoc_pooled.pbs```

#### manifold stats - done

POOLED 25 chunks run on 1-25 chunks: ```qsub -N mf_1000_10 -J 1-25 -l walltime=01:00:00 -l select=1:ncpus=8:mem=32gb -v MANIFEST=metadata_files/syn_data_N1000_P100.csv,CONFIG=configs/manifold_stats.yml,CONFIG_NAME=manifold_stats,N_CHUNKS=25 run_pyspoc_pooled.pbs```

### 10000,100 -- left on CPM counts

#### distance_distrib_stats (done)

50 chunks : ```qsub -J 1-50 -l walltime=01:00:00 -l select=1:ncpus=8:mem=32gb -v MANIFEST=metadata_files/syn_data_N10000_P100.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=distance_distrib_stats,N_CHUNKS=50 run_pyspoc_pooled.pbs```

#### pyspoc internal config

(~50mins for 5 datasets, so 500/5 ~100 chunks)
(10gb not enough mem)
100 chunks, 1-10: ```qsub -N pic_test_10000_100 -J 1-10 -l walltime=02:00:00 -l select=1:ncpus=8:mem=48gb -v MANIFEST=metadata_files/syn_data_N10000_P100.csv,CONFIG=configs/pyspoc_internal_stats.yaml,CONFIG_NAME=pyspoc_internal_stats,N_CHUNKS=100 run_pyspoc_pooled.pbs```
(991)

qsub -N pic_test_10000_100 -J 11-50 -l walltime=02:00:00 -l select=1:ncpus=8:mem=48gb -v MANIFEST=metadata_files/syn_data_N10000_P100.csv,CONFIG=configs/pyspoc_internal_stats.yaml,CONFIG_NAME=pyspoc_internal_stats,N_CHUNKS=100 run_pyspoc_pooled.pbs
(106)

qsub -N pic_test_10000_100 -J 51-100 -l walltime=02:00:00 -l select=1:ncpus=8:mem=48gb -v MANIFEST=metadata_files/syn_data_N10000_P100.csv,CONFIG=configs/pyspoc_internal_stats.yaml,CONFIG_NAME=pyspoc_internal_stats,N_CHUNKS=100 run_pyspoc_pooled.pbs
(283)

#### manifold stats

qsub -N mf_10000_100 -J 1-50 -l walltime=01:00:00 -l select=1:ncpus=8:mem=32gb -v MANIFEST=metadata_files/syn_data_N10000_P100.csv,CONFIG=configs/manifold_stats.yml,CONFIG_NAME=manifold_stats,N_CHUNKS=50 run_pyspoc_pooled.pbs

### 1000,1000 -- left on CPM counts

#### distance_distrib_stats - done

25 chunks on 4-25 chunks: ```qsub -N dist_1000_1000 -J 1-25 -l walltime=00:30:00 -l select=1:ncpus=2:mem=2gb -v MANIFEST=metadata_files/syn_data_N1000_P1000.csv,CONFIG=configs/dist_distrib_stats.yml,CONFIG_NAME=distance_distrib_stats,N_CHUNKS=25 run_pyspoc.pbs```

#### pyspoc internal config - in progress (very long)

25 chunks test on 1-3 chunks: ```qsub -N pic_test_1000_1000 -J 1-3 -l walltime=10:00:00 -l select=1:ncpus=16:mem=48gb -v MANIFEST=metadata_files/syn_data_N1000_P1000.csv,CONFIG=configs/pyspoc_internal_stats.yaml,CONFIG_NAME=pyspoc_internal_stats,N_CHUNKS=25 run_pyspoc.pbs```
(one dataset took 4hrs)

#### manifold stats - done

POOLED 25 chunks run on 1-25 chunks: ```qsub -N mf_1000_1000 -J 1-25 -l walltime=01:00:00 -l select=1:ncpus=4:mem=16gb -v MANIFEST=metadata_files/syn_data_N1000_P1000.csv,CONFIG=configs/manifold_stats.yml,CONFIG_NAME=manifold_stats,N_CHUNKS=25 run_pyspoc_pooled.pbs``

## other helpful commands

```qstat -x -f <jobid> | grep resources_used```

```grep -L 'DONE!' pbs_job_outs/<jobid>*.OU | wc -l```

```grep -h 'STATS FOR INDEX' pbs_job_outs/<pid>*.E*```

```find results/per_dataset/syn_data_N1000_P10/ -name "*.parquet" -printf "%h\n" | sort | uniq -c```

<!-- 
cat metadata_files/syn_data_N10000_P100.csv | wc -l
find results/per_dataset/syn_data_N10000_P100/ -name "*.parquet" -printf "%h\n" | sort | uniq -c -->

tr '\r' '\n' < ERROR_FILE | awk '
function secs(t,  a, n) { n = split(t, a, ":"); return n == 3 ? a[1]*3600 + a[2]*60 + a[3] : a[1]*60 + a[2] }
/Statistic: \[/ { match($0, /\[[^]]+\]/); name = substr($0, RSTART+1, RLENGTH-2) }
/Computing Statistics:/ && match($0, /[0-9]+\/[0-9]+ \[[0-9:]+/) {
    split(substr($0, RSTART, RLENGTH), p, /[\/ \[]+/)
    i = p[1]; t = secs(p[3])
    if (i == 0) { prev = 0; name = ""; print "---- new dataset ----"; next }
    printf "%3d  %8.0fs  %s\n", i, t - prev, (name == "" ? "(unnamed)" : name)
    prev = t; name = ""
} > pbs_job_outs/errlog.txt'

<!-- ```for d in results/per_dataset/syn_data_N1000_P10/*/; do    echo "$d $(ls "$d"*.parquet 2>/dev/null | wc -l)"; done``` -->
