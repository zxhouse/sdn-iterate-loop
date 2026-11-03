"""Closed-loop runs on the live testbed: every test deploys the configuration through
OpenFlow and measures it. The bank means are used only to score what is shipped."""
import sys, json, random, time, numpy as np
import testbed as tb, sdnloop as S, worlds as W
import experiments as X

POL = sys.argv[1].split(",")
SEEDS = [int(s) for s in sys.argv[2].split(",")]
BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else 100
ref = W.testbed_world()
log = open("data/live_tests.jsonl", "a")


class LiveDraw:
    def __init__(self, tag):
        self.tag = tag; self.rng = random.Random(__import__("zlib").crc32(tag.encode()))

    def __call__(self, c, rng):
        x = [(c >> i) & 1 for i in range(tb.NF)]
        r = tb.test(x, 1.5, rng=self.rng); r["tag"] = self.tag
        log.write(json.dumps(r) + "\n"); log.flush()
        return r["u"]


tuned = json.load(open("results/tuned.json"))["high"] if __import__("os").path.exists("results/tuned.json") else {}
out = []
for pol in POL:
    for s in SEEDS:
        tag = f"{pol}#{s}"
        w = S.World(ref.U, LiveDraw(tag), ref.sigma, ref.screen, "live")
        fn, g = X.POLICIES[pol]
        par = X.parse(tuned.get(pol)) if g else None
        if isinstance(par, list): par = tuple(par)
        t0 = time.time()
        o = fn(w, BUDGET, 100000 + s, par)
        o.update({"policy": pol, "seed": s, "budget": BUDGET, "wall": time.time() - t0})
        out.append(o); print(json.dumps(o), flush=True)
        with open("data/live_results.jsonl", "a") as f:
            f.write(json.dumps(o) + "\n")
