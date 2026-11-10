function secs(t, a, n) { n = split(t, a, ":"); return n == 3 ? a[1]*3600 + a[2]*60 + a[3] : a[1]*60 + a[2] }

# last part of a path ("/a/b/c.main" -> "c.main"), trimmed
function base(s, parts, n) {
    gsub(/^ +| +$/, "", s)
    n = split(s, parts, "/")
    return parts[n] != "" ? parts[n] : parts[n-1]
}

# name inside the [...] matched by the calling rule; handles "Stat => Red"
function getname(m, k) {
    m = substr($0, RSTART, RLENGTH)
    m = substr(m, index(m, "[") + 1)
    m = substr(m, 1, length(m) - 1)
    k = index(m, "=>")
    name = k ? base(substr(m, 1, k - 1)) "=>" base(substr(m, k + 2)) : base(m)
    gsub(/ /, "_", name)
}

function progress(phase, p, t, c) {
    if (!match($0, /[0-9]+\/[0-9]+ \[[0-9:]+/)) return
    split(substr($0, RSTART, RLENGTH), p, /[\/ \[]+/); c = p[1] + 0; t = secs(p[3])
    if (phase != curphase || c < last) {          # a new progress bar has started
        curphase = phase; last = -1; prev = 0
        print (phase == "Statistics" ? "---- new dataset ----" : "-- " phase " --")
    }
    if (c <= last) return                         # redraw of the same count: ignore
    last = c
    if (c == 0) return
    printf "%3d %8.0fs  %-18s %s\n", c, t - prev, phase, name == "" ? "(unnamed)" : name
    prev = t; name = ""
}

match($0, /Statistic: \[[^]]+\]/)           { getname() }
match($0, /Reduced Statistic \[[^]]+\]/)    { getname() }
match($0, /Reductions? *:? *\[[^]]+\]/)     { getname() }

/Computing Statistics:/         { progress("Statistics") }
/Computing Reduced Statistics:/ { progress("Reduced Statistics") }
/Computing Reductions:/         { progress("Reductions") }