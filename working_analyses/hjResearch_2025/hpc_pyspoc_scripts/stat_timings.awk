function secs(t, a, n) { n = split(t, a, ":"); return n == 3 ? a[1]*3600 + a[2]*60 + a[3] : a[1]*60 + a[2] }
/Statistic: \[/ { match($0, /\[[^]]+\]/); name = substr($0, RSTART+1, RLENGTH-2) }
/Computing Statistics:/ && match($0, /[0-9]+\/[0-9]+ \[[0-9:]+/) {
    split(substr($0, RSTART, RLENGTH), p, /[\/ \[]+/); t = secs(p[3])
    if (p[1] == 0) { prev = 0; name = ""; print "---- new dataset ----"; next }
    printf "%3d %8.0fs  %s\n", p[1], t - prev, name == "" ? "(unnamed)" : name; prev = t; name = ""
}