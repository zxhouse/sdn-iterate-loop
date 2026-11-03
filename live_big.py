"""Closed-loop runs on the live GEANT testbed; bank means are used only for scoring."""
import sys, json, random, time, zlib, os
import bigbed as B, sdnloop as S, worlds as W, experiments as X
POL = sys.argv[1].split(","); SEEDS = [int(s) for s in sys.argv[2].split(",")]
BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else 100
LEVEL = sys.argv[4] if len(sys.argv) > 4 else "mid"
ref = W.big_world(); log = open("data/big_live_tests.jsonl", "a")
done = set()
if os.path.exists("data/big_live_results.jsonl"):
    for l in open("data/big_live_results.jsonl"):
        d = json.loads(l); done.add((d["policy"], d["seed"]))


class LiveDraw:
    def __init__(self, tag):
        self.tag = tag; self.rng = random.Random(zlib.crc32(tag.encode()))

    def __call__(self, c, rng):
        r = B.test([(c >> i) & 1 for i in range(B.NF)], rng=self.rng); r["tag"] = self.tag
        log.write(json.dumps(r) + "\n"); log.flush(); return r["u"]


tuned = json.load(open("results/tuned.json"))[LEVEL]
for pol in POL:
    for s in SEEDS:
        if (pol, s) in done: continue
        w = S.World(ref.U, LiveDraw(f"{pol}#{s}"), ref.sigma, ref.screen, "live-geant")
        fn, g = X.POLICIES[pol]; par = X.parse(tuned.get(pol)) if g else None
        if isinstance(par, list): par = tuple(par)
        t0 = time.time(); o = fn(w, BUDGET, 100000 + s, par)
        o.update({"policy": pol, "seed": s, "budget": BUDGET, "wall": time.time() - t0, "level": LEVEL})
        print(json.dumps(o), flush=True)
        with open("data/big_live_results.jsonl", "a") as f: f.write(json.dumps(o) + "\n")
