"""Cross-validated replay of all twelve policies on the small testbed (tuned at the testbed's noise level)."""
import json, numpy as np
import worlds as W
from split_full import run
T = json.load(open("results/tuned.json"))
s = [([0, 1, 2], [3, 4]), ([2, 3, 4], [0, 1]), ([0, 2, 4], [1, 3]), ([1, 3, 4], [0, 2])]
out = json.load(open("results/split_full.json")) if __import__("os").path.exists("results/split_full.json") else {}
out["small"] = {"high": run("data/bank.jsonl", W.testbed_twin(), s, T["high"])}
print("small", {p: round(v["mean"], 1) for p, v in out["small"]["high"].items()}, flush=True)
json.dump(out, open("results/split_small.json", "w"), indent=1)
