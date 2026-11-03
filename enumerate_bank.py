"""Exhaustive measurement bank: every configuration measured once per pass, random order per pass."""
import testbed as tb, json, random, sys, os, itertools, time
passes = [int(p) for p in sys.argv[1].split(",")]
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
out = "data/bank.jsonl"
done = set()
if os.path.exists(out):
    for l in open(out):
        d = json.loads(l); done.add((d["pass"], d["x"]))
allx = ["".join(map(str, b)) for b in itertools.product([0, 1], repeat=tb.NF)]
for p in passes:
    rng = random.Random(1000 + p)
    order = allx[:]; rng.shuffle(order)
    t0 = time.time()
    with open(out, "a") as f:
        for i, xs in enumerate(order):
            if (p, xs) in done: continue
            r = tb.test([int(c) for c in xs], dur, rng=rng)
            r["pass"] = p
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 100 == 0: print(p, i, round(time.time() - t0), flush=True)
