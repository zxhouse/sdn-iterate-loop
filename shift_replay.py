"""Replay with a configuration-specific shift between the bank and the live network.
Each test returns a bank measurement plus a fixed per-configuration offset d(p) ~ N(0, s)
plus extra noise; what is shipped is still scored against the bank means. s and the extra
noise are the values estimated from the live runs (results/live_analysis.json)."""
import json, math, numpy as np
import worlds as W, sdnloop as S, experiments as X

POLS = ["loop", "ucb_screen", "measure_once"]


def shifted_world(base, smp, s_shift, extra_sd, seed):
    d = np.random.default_rng(seed).normal(0, s_shift, len(smp))
    def draw(c, rng):
        return float(smp[c][rng.integers(len(smp[c]))]) + d[c] + (rng.normal(0, extra_sd) if extra_sd else 0.0)
    return S.World(base.U, draw, float(math.hypot(base.sigma, extra_sd)), base.screen, "shifted")


def run(name, tuned, n_shift=40, runs_per=10):
    A = json.load(open("results/live_analysis.json"))[name]
    base = W.testbed_world() if name == "small" else W.big_world()
    smp = W.testbed_samples() if name == "small" else W.big_samples()
    extra = math.sqrt(max(A["live_sd_ge4"] ** 2 - A["bank_sd"] ** 2, 0.0))
    out = {"s_shift": A["shift_excess_sd"], "extra_sd": extra}
    for s_shift in (A["shift_excess_sd"], 0.0):
        key = "with_shift" if s_shift > 0 else "noise_only"
        out[key] = {}
        for p in POLS:
            fn, g = X.POLICIES[p]; par = X.parse(tuned.get(p)) if g else None
            if isinstance(par, list): par = tuple(par)
            vals = []
            for k in range(n_shift):
                w = shifted_world(base, smp, s_shift, extra, 7000 + k)
                vals += [fn(w, 100, 100000 + 1000 * k + r, par)["shipped"] for r in range(runs_per)]
            out[key][p] = {"mean": 100 * float(np.mean(vals)), "hw": 100 * 1.96 * float(np.std(vals, ddof=1)) / math.sqrt(len(vals))}
        print(name, key, {p: round(v["mean"], 2) for p, v in out[key].items()}, flush=True)
    return out


if __name__ == "__main__":
    T = json.load(open("results/tuned.json"))["high"]
    res = {"geant": run("geant", T), "small": run("small", T)}
    json.dump(res, open("results/shift_replay.json", "w"), indent=1)
