#!/usr/bin/env bash
# usage: bash summarise_runs.sh submit.log [LINES] [folder_with_ER_files]
#   LINES:  all (default) | -6 (last 6 lines) | 12 (line 12) | 10-15 (lines 10 to 15)
set -uo pipefail
LOG=$1
SEL=${2:-all}
DIR=${3:-.}
AWK=stat_timings.awk
OUT_EXT=OU                           # extension of the PBS output files, e.g. <PID>[1].pbs-7.OU

select_lines() {
    case "$SEL" in
        all)  cat "$LOG" ;;
        -*)   tail -n "${SEL#-}" "$LOG" ;;
        *-*)  sed -n "${SEL%-*},${SEL#*-}p" "$LOG" ;;
        *)    sed -n "${SEL}p" "$LOG" ;;
    esac
}

# errorParse output -> rows: job, dataset, phase, statistic, seconds
to_rows() {
    awk -v job="$1" '
        /^---- new dataset/       { ds++; phase = "Statistics"; next }
        /^-- Reductions/          { phase = "Reductions"; next }
        /^-- Reduced Statistics/  { phase = "Reduced Statistics"; next }
        $2 ~ /^[0-9]+s$/ { t = $2; sub(/s$/, "", t); print job "\t" ds "\t" phase "\t" $NF "\t" t }'
}

# PBS usage block -> row: job, cpu_s, cpu_pct, mem_kb, wall_s
to_usage() {
    awk -v job="$1" -F': ' '
        function s(t, a) { split(t, a, ":"); return a[1]*3600 + a[2]*60 + a[3] }
        /CPU Time used/   { cpu = s($2) }
        /CPU Percent/     { pct = $2 + 0 }
        /Memory usage/    { mem = $2 + 0 }
        /Walltime usage/  { wall = s($2) }
        END { if (wall != "") print job "\t" cpu "\t" pct "\t" mem "\t" wall }' "$2"
}

# stats for one column of the usage table
colstat() {
    cut -f"$2" "$1" | sort -n | awk -v label="$3" -v kind="$4" '
        function hms(x) { return sprintf("%02d:%02d:%02d", x/3600, (x%3600)/60, x%60) }
        function fmt(x) { return kind == "time" ? hms(x) : kind == "mem" ? sprintf("%.2f GB", x/1048576) : sprintf("%.0f%%", x) }
        { v[++n] = $1; s += $1 }
        END {
            if (!n) exit
            med = n % 2 ? v[(n+1)/2] : (v[n/2] + v[n/2+1]) / 2
            printf "  %-10s mean %-10s median %-10s min %-10s max %-10s\n", label, fmt(s/n), fmt(med), fmt(v[1]), fmt(v[n])
        }'
}

select_lines | while IFS='|' read -r ts jobid rest; do
    jobid=$(sed 's/^ *//;s/ *$//' <<< "$jobid")
    rest=$(sed 's/^ *//;s/ *$//' <<< "$rest")
    [ -z "$jobid" ] && continue

    pid=${jobid%%\[*}                # part before "["
    suffix=${jobid#*\]}              # part after "]", e.g. ".pbs-7"
    config_name=${rest%% *}          # first word
    range=$(grep -o 'array=[0-9-]*' <<< "$rest" | cut -d= -f2)
    first=${range%-*}; last=${range#*-}

    out="${pid}_${config_name}.txt"
    raw=$(mktemp); res=$(mktemp); usg=$(mktemp)

    {
        echo "# $ts| $jobid | $rest"
        for i in $(seq "$first" "$last"); do
            f="$DIR/${pid}[$i]${suffix}.ER"
            o="$DIR/${pid}[$i]${suffix}.${OUT_EXT}"

            if [ -f "$o" ]; then to_usage "$i" "$o" >> "$usg"
            else echo "## array $i: MISSING $o"; fi

            if [ ! -f "$f" ]; then echo "## array $i: MISSING $f"; continue; fi
            rows=$(awk -f "$AWK" "$f" | to_rows "$i")

            line=$(grep -h "results saved out of" "$f" | tail -n 1 |
                   sed -nE 's/.*: ([0-9]+) results saved out of ([0-9]+) total; ([0-9]+) skipped.*/\1 \2 \3/p')
            json=$(grep -h "JSON error files" "$f" | tail -n 1 |
                   sed -nE 's/.*: ([0-9]+) JSON error files.*/\1/p')

            if [ -n "$line" ]; then
                echo "$line ${json:-NA}" >> "$res"
                [ -n "$rows" ] && echo "$rows" >> "$raw"
            else
                reason=$(grep -hi "exceeded limit" "$f" "$o" 2>/dev/null | head -n 1)
                echo "## array $i: incomplete (${reason:-no final STATS line}); last dataset excluded from timings"
                if [ -n "$rows" ]; then
                    lastds=$(cut -f2 <<< "$rows" | sort -n | tail -n 1)
                    awk -F'\t' -v d="$lastds" '$2 != d' <<< "$rows" >> "$raw"
                fi
            fi
        done
        echo
        echo "# resource usage per array job ($(wc -l < "$usg") jobs)"
        colstat "$usg" 5 "Walltime" time
        colstat "$usg" 2 "CPU time" time
        colstat "$usg" 3 "CPU %"    pct
        colstat "$usg" 4 "Memory"   mem

        echo
        echo "# results across array jobs"
        awk '
            { n++; s += $1; t += $2; k += $3
              if (n == 1 || $1 < mn) mn = $1
              if ($1 > mx) mx = $1
              if ($4 != "NA") { jn++; j += $4; if ($4 > 0) jbad++ } }
            END {
                if (!n) { print "  no STATS lines found"; exit }
                printf "  jobs with results: %d\n", n
                printf "  saved: %d of %d total (%.1f%%), skipped/pre-computed: %d\n", s, t, (t ? 100*s/t : 0), k
                printf "  saved per job: mean %.1f, min %d, max %d\n", s/n, mn, mx
                if (jn) printf "  JSON error files: %d total, in %d of %d jobs\n", j, jbad + 0, jn
                else    print  "  JSON error files: no lines found"
            }' "$res"

        echo
        echo "# timing summary (seconds) across all array jobs"
        awk -F'\t' '
            {
                o = ($3 == "Statistics") ? 1 : ($3 == "Reductions") ? 2 : 3
                print o " " $3 "\t" $4 "\t" $5
                total[$1 "_" $2] += $5
            }
            END { for (d in total) print "0 Total per dataset\tall\t" total[d] }
        ' "$raw" |
        sort -t$'\t' -k1,1 -k2,2 -k3,3n |
        awk -F'\t' '
            function flush() {
                if (!n) return
                med = n % 2 ? v[(n+1)/2] : (v[n/2] + v[n/2+1]) / 2
                printf "  %-83s %4d %9.1f %9.1f %8d %8d\n", cur, n, sum/n, med, v[1], v[n]
            }
            BEGIN { printf "  %-83s %4s %9s %9s %8s %8s\n", "statistic", "n", "mean", "median", "min", "max" }
            $1 != ph { flush(); n = 0; cur = ""; ph = $1; print "\n" substr(ph, 3) }
            $2 != cur { flush(); cur = $2; n = 0; sum = 0 }
            { v[++n] = $3; sum += $3 }
            END { flush() }'
    } > "$out"

    rm -f "$raw" "$res" "$usg"
    echo "wrote $out"
done