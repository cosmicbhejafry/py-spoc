what does this folder have?

contains scripts to run pyspoc on different datasets and config files in a batched-way on the hpc

the full file tree looks like this currently (some folders have been .gitignored on the repo)
hpc_pyspoc_scripts/
├── configs #contains config YMLs
├── concat_outs.sh #script to concat outputs
├── metadata_files #contains .csv 's with information about datasets to run pyspoc on
├── pbs_job_outs # .o / .e job outs
├── py_scripts #python scripts to run and concat outs
├── README.md 
├── results #store results
└── run_pyspoc.sh #script to run pyspoc-array job

assuming: 
- metadata_files stay untouched/unedited during a full run

---

TODO: ask G about preprocessing. 
currently even the counts are going through a standardscaler - that needs to change

# helpful commands

```qstat -x -f <jobid> | grep resources_used```

```grep -L 'DONE!' pbs_job_outs/<jobid>*.OU | wc -l```

```grep -h 'STATS FOR INDEX' pbs_job_outs/<pid>*.E*```

```find results/per_dataset/syn_data_N1000_P10/ -name "*.parquet" -printf "%h\n" | sort | uniq -c```

<!-- 
cat metadata_files/syn_data_N10000_P100.csv | wc -l
find results/per_dataset/syn_data_N10000_P100/ -name "*.parquet" -printf "%h\n" | sort | uniq -c -->

tr '\r' '\n' < ERRFILE | awk '
function secs(t,  a, n) { n = split(t, a, ":"); return n == 3 ? a[1]*3600 + a[2]*60 + a[3] : a[1]*60 + a[2] }
/Statistic: \[/ { match($0, /\[[^]]+\]/); name = substr($0, RSTART+1, RLENGTH-2) }
/Computing Statistics:/ && match($0, /[0-9]+\/[0-9]+ \[[0-9:]+/) {
    split(substr($0, RSTART, RLENGTH), p, /[\/ \[]+/)
    i = p[1]; t = secs(p[3])
    if (i == 0) { prev = 0; name = ""; print "---- new dataset ----"; next }
    printf "%3d  %8.0fs  %s\n", i, t - prev, (name == "" ? "(unnamed)" : name)
    prev = t; name = ""
} > pbs_job_outs/errlog.txt'

OR: 
tr '\r' '\n' < pbs_job_outs/<pid>.pbs-7.ER | awk -f stat_timings.awk > pbs_job_outs/err_logs_awks/errlog_gtex_N200_P500.txt

tr '\r' '\n' < pbs_job_outs/4581400\[2\].pbs-7.ER | awk -f stat_timings.awk > pbs_job_outs/err_logs_awks/errlog_new_sts.txt

---

'fast_stats_graph..' --> removed Kmedoids and MCG

<!-- ```for d in results/per_dataset/syn_data_N1000_P10/*/; do    echo "$d $(ls "$d"*.parquet 2>/dev/null | wc -l)"; done``` -->
