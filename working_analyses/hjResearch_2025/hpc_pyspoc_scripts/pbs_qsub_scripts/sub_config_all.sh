#!/usr/bin/env bash
# usage: bash submit_config.sh <manifest_name> <idx|all> [test|full]
#        bash submit_config.sh <manifest_name> <idx> rerun <start-end>      e.g. rerun 17-23

# examples:
# bash submit_config.sh syn_data_N1000_P100_10seeds 3 test
# bash submit_config.sh syn_data_N1000_P100_10seeds all
# bash submit_config.sh syn_data_N1000_P100_10seeds 3 rerun 17-23

# RUNS FROM hpc_pyspoc_scripts/
set -eo pipefail

USAGE="usage: bash submit_config.sh <manifest_name> <idx|all> [test|full|rerun <start-end>]"
MANIFEST_NAME=${1:?$USAGE}
WHICH=${2:?$USAGE}
MODE=${3:-full}
RERUN=${4:-}
MANIFEST=metadata_files/${MANIFEST_NAME}.csv
RES=pbs_qsub_scripts/resource_files/pbsParams_${MANIFEST_NAME}.csv

[ -f "$MANIFEST" ] || { echo "missing $MANIFEST . run from hpc_pyspoc_scripts/" >&2; exit 1; }
[ -f "$RES" ]      || { echo "missing $RES" >&2; exit 1; }
if [ "$MODE" = "rerun" ]; then
    [ -n "$RERUN" ]       || { echo "rerun needs a range, e.g. rerun 17-23" >&2; exit 1; }
    [ "$WHICH" != "all" ] || { echo "rerun works on one config at a time, not 'all'" >&2; exit 1; }
fi

submit() {
    local i=$1 row
    row=$(awk -F, -v i="$i" 'NR > 1 && !/^#/ && $1 == i' "$RES")
    row=${row%$'\r'}                                   # drop Windows line ending, if any
    [ -n "$row" ] || { echo "no row for $i in $RES" >&2; exit 1; }
    IFS=, read -r _ CONFIG CONFIG_NAME N_CHUNKS WALLTIME NCPUS MEM NGPUS PBS_SCRIPT_NAME <<< "$row"

    local GPU_REQ=""
    [ "${NGPUS:-0}" -gt 0 ] && GPU_REQ=":ngpus=$NGPUS"
    local VARS="MANIFEST=$MANIFEST,CONFIG=$CONFIG,CONFIG_NAME=$CONFIG_NAME,N_CHUNKS=$N_CHUNKS"

    # rerun: user-given array range, e.g. 17-23
    if [ "$MODE" = "rerun" ]; then
        [[ $RERUN =~ ^([0-9]+)-([0-9]+)$ ]] || { echo "rerun range must look like 17-23" >&2; exit 1; }
        local a=${BASH_REMATCH[1]} b=${BASH_REMATCH[2]}
        if [ "$a" -lt 1 ] || [ "$b" -gt "$N_CHUNKS" ] || [ "$a" -ge "$b" ]; then
            echo "bad range $RERUN: need 1 <= start < end <= $N_CHUNKS" >&2; exit 1
        fi
        id=$(qsub -J "$RERUN" -l walltime="$WALLTIME" -l select=1:ncpus=$NCPUS:mem=$MEM$GPU_REQ \
                  -v "$VARS" "$PBS_SCRIPT_NAME")
        echo "$(date '+%F %T') | $id | $CONFIG_NAME | $MANIFEST_NAME | array=$RERUN n_chunks=$N_CHUNKS ncpus=$NCPUS mem=$MEM walltime=$WALLTIME mode=rerun" \
            | tee -a qsub_history.log
        return
    fi

    local ranges
    if [ "$MODE" = "test" ]; then
        #FIRST 2 ARRAY JOBS SUB
        ranges="1-2"
    else
        # DEFINE RANGES AS: 1-50, 51-100, 101-... UNTIL N_CHUNKS
        ranges=$(for s in $(seq 1 50 "$N_CHUNKS"); do
                     echo "$s-$(( s + 49 < N_CHUNKS ? s + 49 : N_CHUNKS ))"
                 done)
    fi

    for ARRAY in $ranges; do
        # QSUB COMMAND
        id=$(qsub -J "$ARRAY" -l walltime="$WALLTIME" -l select=1:ncpus=$NCPUS:mem=$MEM$GPU_REQ \
                  -v "$VARS" "$PBS_SCRIPT_NAME")
        # ECHO IN LOG
        echo "$(date '+%F %T') | $id | $CONFIG_NAME | $MANIFEST_NAME | array=$ARRAY n_chunks=$N_CHUNKS ncpus=$NCPUS mem=$MEM walltime=$WALLTIME mode=$MODE" \
            | tee -a qsub_history.log
    done
}

if [ "$WHICH" = "all" ]; then
    for i in $(awk -F, 'NR > 1 && !/^#/ && NF {print $1}' "$RES"); do submit "$i"; done
else
    submit "$WHICH"
fi