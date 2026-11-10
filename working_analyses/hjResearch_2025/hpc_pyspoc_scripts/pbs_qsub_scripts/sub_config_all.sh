#!/usr/bin/env bash
# usage: bash submit_config.sh <manifest_name> <idx|all> [test]
# RUNS FROM hpc_pyspoc_scripts/
set -eo pipefail

MANIFEST_NAME=${1:?usage: bash submit_config.sh <manifest_name> <idx|all> [test]}
WHICH=${2:?usage: bash submit_config.sh <manifest_name> <idx|all> [test]}
MODE=${3:-full}
MANIFEST=metadata_files/${MANIFEST_NAME}.csv
RES=pbs_qsub_scripts/resource_files/pbsParams_${MANIFEST_NAME}.csv

[ -f "$MANIFEST" ] || { echo "missing $MANIFEST . run from hpc_pyspoc_scripts/" >&2; exit 1; }
[ -f "$RES" ]      || { echo "missing $RES" >&2; exit 1; }

submit() {
    local i=$1 row
    row=$(awk -F, -v i="$i" 'NR > 1 && !/^#/ && $1 == i' "$RES")
    row=${row%$'\r'}                                   # drop Windows line ending, if any
    [ -n "$row" ] || { echo "no row for $i in $RES" >&2; exit 1; }
    IFS=, read -r _ CONFIG CONFIG_NAME N_CHUNKS WALLTIME NCPUS MEM NGPUS PBS_SCRIPT_NAME <<< "$row"

    local ranges
    if [ "$MODE" = "test" ]; then
        ranges="1-2"
    else
        ranges=$(for s in $(seq 1 50 "$N_CHUNKS"); do
                     echo "$s-$(( s + 49 < N_CHUNKS ? s + 49 : N_CHUNKS ))"
                 done)
    fi

    local GPU_REQ=""
    [ "${NGPUS:-0}" -gt 0 ] && GPU_REQ=":ngpus=$NGPUS"

    for ARRAY in $ranges; do
        id=$(qsub -J "$ARRAY" -l walltime="$WALLTIME" -l select=1:ncpus=$NCPUS:mem=$MEM$GPU_REQ \
                  -v MANIFEST=$MANIFEST,CONFIG=$CONFIG,CONFIG_NAME=$CONFIG_NAME,N_CHUNKS=$N_CHUNKS \
                  "$PBS_SCRIPT_NAME")
        echo "$(date '+%F %T') | $id | $CONFIG_NAME | $MANIFEST_NAME | array=$ARRAY n_chunks=$N_CHUNKS ncpus=$NCPUS mem=$MEM walltime=$WALLTIME mode=$MODE" \
            | tee -a qsub_history.log
    done
}

if [ "$WHICH" = "all" ]; then
    for i in $(awk -F, 'NR > 1 && !/^#/ && NF {print $1}' "$RES"); do submit "$i"; done
else
    submit "$WHICH"
fi