"""Exhaustive measurement bank on the GEANT testbed (resumable)."""
import bigbed as B, json, random, sys, os, itertools, time
passes = [int(p) for p in sys.argv[1].split(",")]
out = "data/big_bank.jsonl"
done = set()
if os.path.exists(out):
    for l in open(out):
        try: d = json.loads(l); done.add((d["pass"], d["x"]))
        except Exception: pass
allx = ["".join(map(str, b)) for b in itertools.product([0, 1], repeat=B.NF)]
for p in passes:
    rng = random.Random(2000 + p); order = allx[:]; rng.shuffle(order); t0 = time.time()
    with open(out, "a") as f:
        for i, xs in enumerate(order):
            if (p, xs) in done: continue
            r = B.test([int(c) for c in xs], rng=rng); r["pass"] = p
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 100 == 0: print(p, i, round(time.time() - t0), flush=True)
