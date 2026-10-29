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