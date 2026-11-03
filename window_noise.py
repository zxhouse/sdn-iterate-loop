"""Test noise as a function of the measurement window (E1b)."""
import testbed as tb, json, random, sys
cfgs = sys.argv[1].split(","); wins = [1.0, 2.0, 3.0, 5.0]; reps = int(sys.argv[2])
rng = random.Random(4242)
with open("data/window_noise.jsonl", "a") as f:
    for r in range(reps):
        for xs in cfgs:
            for wd in wins:
                d = tb.test([int(c) for c in xs], wd, rng=rng); d["rep"] = r
                f.write(json.dumps(d) + "\n"); f.flush()
        print("rep", r, flush=True)
