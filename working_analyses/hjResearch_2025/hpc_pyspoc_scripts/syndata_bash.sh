#!/bin/bash
# usage: bash submit_config.sh <1|2|3|all> [test]
set -eo pipefail

MANIFEST=metadata_files/syn_data_N1000_P100_10seeds.csv
MODE=${2:-full}

submit() {
    local i=$1
    case $i in
        1) CONFIG=configs/dist_distrib_stats_metricSweep.yml; CONFIG_NAME=dist_distrib_stats
           N_CHUNKS=100;  WALLTIME=00:10:00; NCPUS=1; MEM=2gb; PBS_SCRIPT_NAME=run_pyspoc.pbs ;;
        2) CONFIG=configs/manifold_stats_ID.yml; CONFIG_NAME=manifold_stats
           N_CHUNKS=200; WALLTIME=01:00:00; NCPUS=4; MEM=6gb; PBS_SCRIPT_NAME=run_pyspoc_pooled.pbs ;;
        3) CONFIG=configs/pyspoc_internal_stats_fast.yaml; CONFIG_NAME=pyspoc_internal
           N_CHUNKS=50; WALLTIME=06:30:00; NCPUS=2; MEM=6gb; PBS_SCRIPT_NAME=run_pyspoc_pooled.pbs ;;
        *) echo "unknown config: $i" >&2; exit 1 ;;
    esac

    local ranges
    if [ "$MODE" = "test" ]; then
        ranges="1-2"
    else
        ranges=$(for s in $(seq 1 50 "$N_CHUNKS"); do
                     echo "$s-$(( s + 49 < N_CHUNKS ? s + 49 : N_CHUNKS ))"
                 done)
    fi

    for ARRAY in $ranges; do
        id=$(qsub -J "$ARRAY" -l walltime="$WALLTIME" -l select=1:ncpus=$NCPUS:mem=$MEM \
             -v MANIFEST=$MANIFEST,CONFIG=$CONFIG,CONFIG_NAME=$CONFIG_NAME,N_CHUNKS=$N_CHUNKS \
             $PBS_SCRIPT_NAME)
        echo "$(date '+%F %T') | $id | $CONFIG_NAME array=$ARRAY n_chunks=$N_CHUNKS ncpus=$NCPUS mem=$MEM mode=$MODE" \
            | tee -a qsub_history.log
        # RESULTS_DIR=$EPHEMERAL/pyspoc_runs/results/per_dataset/$(basename "$MANIFEST" .csv)/$(basename "$CONFIG" .yml)
        # qsub -W depend=afterany:$id -v JOBID=$id,CONFIG_NAME=$CONFIG_NAME,RESULTS_DIR=$RESULTS_DIR check_job.pbs    
    done
}

if [ "${1:-}" = "all" ]; then
    for i in 1 3 2; do submit "$i"; done
else
    submit "${1:?usage: bash submit_config.sh <1|2|3|all> [test]}"
fi