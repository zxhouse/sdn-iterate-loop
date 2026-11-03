"""Analysis of the closed-loop runs against the replay (E10, E11): differences between
policies, seed-matched replay, replay with the live noise level, configuration-specific
shifts between the bank and the live runs, and transients after a configuration change."""
import json, math, numpy as np
import worlds as W, experiments as X
from scipy import stats

POLS = ["loop", "ucb_screen", "measure_once"]


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = a.mean() - b.mean(); va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb); df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return {"diff": 100 * d, "ci95": 100 * stats.t.ppf(0.975, df) * se, "p": float(2 * stats.t.sf(abs(d) / se, df))}


def perm_p(a, b, n=20000, seed=0):
    a, b = np.asarray(a), np.asarray(b); obs = a.mean() - b.mean(); z = np.concatenate([a, b])
    rng = np.random.default_rng(seed); c = 0
    for _ in range(n):
        rng.shuffle(z); c += abs(z[:len(a)].mean() - z[len(a):].mean()) >= abs(obs) - 1e-12
    return c / n


def analyse(name, world, live_res, live_tests, replay, tuned, extra_sd):
    us = world.ustar if hasattr(world, "ustar") else world.U.max()
    R = [json.loads(l) for l in open(live_res)]
    T = [json.loads(l) for l in open(live_tests)]
    smp = W.testbed_samples() if name == "small" else W.big_samples()
    bank_mean = np.array([s.mean() for s in smp]); bank_sd = np.sqrt(np.mean([s.var(ddof=1) for s in smp]))
    out = {"policies": {}, "pairs": {}}
    live = {p: [r["shipped"] for r in R if r["policy"] == p] for p in POLS}
    n = len(live["loop"])
    for p in POLS:
        rep = np.array(replay[p]["shipped"]); sm = rep[:n]          # replay seeds 100000+s, s = 0..n-1
        z = (np.mean(live[p]) - rep.mean()) / (rep.std(ddof=1) / math.sqrt(n))
        out["policies"][p] = {"live": 100 * np.mean(live[p]), "live_hw": 100 * stats.t.ppf(0.975, n - 1) * np.std(live[p], ddof=1) / math.sqrt(n),
                              "replay": 100 * rep.mean(), "replay_seedmatched": 100 * sm.mean(),
                              "replay_sd_run": 100 * rep.std(ddof=1), "z_live_vs_replay": z,
                              "p_live_vs_replay": float(2 * stats.norm.sf(abs(z)))}
    for p in POLS[1:]:
        out["pairs"]["loop-" + p] = welch(live["loop"], live[p]) | {"p_perm": perm_p(live["loop"], live[p])}
    # smallest difference detectable with 80% power, two-sided 5%, given the live spread
    sd_pool = np.sqrt(np.mean([np.var(live[p], ddof=1) for p in POLS]))
    out["mde80"] = 100 * (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * sd_pool * math.sqrt(2 / n)
    out["runs_for_1pt"] = int(math.ceil(2 * ((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * sd_pool / 0.01) ** 2))
    # replay with the noise of the live runs
    wl = (W.testbed_world if name == "small" else W.big_world)(extra_sd=extra_sd)
    out["replay_live_noise"] = {}
    for p in POLS:
        fn, g = X.POLICIES[p]; par = X.parse(tuned.get(p)) if g else None
        if isinstance(par, list): par = tuple(par)
        res = [fn(wl, 100, 100000 + s, par)["shipped"] for s in range(400)]
        out["replay_live_noise"][p] = 100 * float(np.mean(res))
    # live tests against the bank: overall residual, configuration-specific shift, transients
    codes = np.array([W.code_of(t["x"]) for t in T]); u = np.array([t["u"] for t in T])
    res = u - bank_mean[codes]
    out["residual_mean"], out["residual_sd"] = float(res.mean()), float(res.std(ddof=1))
    by = {}
    for c, r, uu in zip(codes, res, u): by.setdefault(int(c), []).append((r, uu))
    nb = len(smp[0])
    shift, expv, live_sd = [], [], []
    for c, v in by.items():
        if len(v) >= 4:
            rr = np.array([a for a, _ in v]); uu = np.array([b for _, b in v])
            shift.append(rr.mean()); live_sd.append(uu.std(ddof=1))
            expv.append(uu.var(ddof=1) / len(v) + smp[c].var(ddof=1) / nb)
    shift = np.array(shift)
    out["n_configs_ge4"] = len(shift)
    out["live_sd_ge4"] = float(np.sqrt(np.mean(np.square(live_sd))))
    out["bank_sd"] = float(bank_sd)
    out["shift_sd_observed"] = float(shift.std(ddof=1)); out["shift_sd_expected_noise"] = float(np.sqrt(np.mean(expv)))
    out["shift_excess_sd"] = float(np.sqrt(max(shift.var(ddof=1) - np.mean(expv), 0.0)))
    # transient: test of a configuration that differs from the previous test of the same run vs consecutive re-test
    tags = [t["tag"] for t in T]; changed = []
    for i in range(len(T)):
        changed.append(i == 0 or tags[i] != tags[i - 1] or codes[i] != codes[i - 1])
    changed = np.array(changed)
    out["transient"] = {"after_change_mean": float(res[changed].mean()), "after_change_n": int(changed.sum()),
                        "repeat_mean": float(res[~changed].mean()) if (~changed).any() else None, "repeat_n": int((~changed).sum())}
    if (~changed).sum() > 1:
        out["transient"]["diff_p"] = float(stats.ttest_ind(res[changed], res[~changed], equal_var=False).pvalue)
    return out


if __name__ == "__main__":
    T = json.load(open("results/tuned.json"))
    small = analyse("small", W.testbed_world(), "data/live_results.jsonl", "data/live_tests.jsonl",
                    json.load(open("results/policies_testbed.json"))["100"], T["high"], extra_sd=math.sqrt(0.082 ** 2 - 0.076 ** 2))
    big = analyse("geant", W.big_world(), "data/big_live_results.jsonl", "data/big_live_tests.jsonl",
                  json.load(open("results/big_policies.json"))["high"]["100"], T["high"], extra_sd=math.sqrt(0.076 ** 2 - 0.068 ** 2))
    json.dump({"small": small, "geant": big}, open("results/live_analysis.json", "w"), indent=1, default=float)
    print(json.dumps({"small": small, "geant": big}, indent=1, default=float))
