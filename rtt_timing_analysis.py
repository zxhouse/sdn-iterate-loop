"""Analysis of data/rtt_timing.jsonl: share of probes sent after the transfers ended, and the
effect of averaging the RTT over the whole probe window rather than the loaded part only."""
import json, numpy as np
from scipy import stats
import testbed as tb

R = [json.loads(l) for l in open("data/rtt_timing.jsonl")]
frac, d_all, d_load, after_rtt, L_all, L_load, base = [], [], [], [], [], [], []
for r in R:
    rep = r["replies"]
    if not rep: continue
    s = np.array([x[1] for x in rep]); rt = np.array([x[2] for x in rep])
    end = r["tcp_end"]; ld = s <= end
    frac.append(1 - ld.mean())
    ra = rt.mean(); rl = rt[ld].mean() if ld.any() else ra
    d_all.append(ra); d_load.append(rl)
    if (~ld).any(): after_rtt.append(rt[~ld].mean())
    L_all.append(max(0, 1 - ra / tb.RTT_MAX)); L_load.append(max(0, 1 - rl / tb.RTT_MAX))
frac, d_all, d_load, L_all, L_load = map(np.array, (frac, d_all, d_load, L_all, L_load))
dl = (L_all - L_load) * 0.25            # change of the utility (weight of L)
# configuration level: average the two repetitions of each configuration
by = {}
for r, a, b in zip(R, L_all, L_load): by.setdefault(r["x"], []).append((a, b))
ca = np.array([np.mean([a for a, _ in v]) for v in by.values()]); cb = np.array([np.mean([b for _, b in v]) for v in by.values()])
out = {"tests": len(frac), "share_after_mean": float(frac.mean()), "share_after_q10_q90": [float(np.quantile(frac, .1)), float(np.quantile(frac, .9))],
       "probes_after_mean": float(frac.mean() * 15),
       "rtt_all_mean": float(d_all.mean()), "rtt_loaded_mean": float(d_load.mean()), "rtt_after_mean": float(np.mean(after_rtt)),
       "L_all_mean": float(L_all.mean()), "L_loaded_mean": float(L_load.mean()),
       "dU_mean": float(dl.mean()), "dU_sd": float(dl.std(ddof=1)), "dU_abs_q90": float(np.quantile(np.abs(dl), .9)),
       "pearson_L_tests": float(np.corrcoef(L_all, L_load)[0, 1]), "spearman_L_configs": float(stats.spearmanr(ca, cb)[0]),
       "pearson_L_configs": float(np.corrcoef(ca, cb)[0, 1]), "configs": len(ca)}
json.dump(out, open("results/rtt_timing.json", "w"), indent=1); print(json.dumps(out, indent=1))
