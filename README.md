# Prototype, Test, Re-Test: the iterate–test loop for SDN configuration

Companion code and data for the article

> Kuraś, P.; Bolanowski, M.; Biernacka, E.; Organiściak, P.; Turek, A.; Gerka, A.; Gacek, M.; Łazarski, M.; Misiura, F.; Michura-Mika, K. *Prototype, Test, Re-Test: Convergence Conditions and Testbed Evidence for Design-Thinking Iteration in Software-Defined Network Configuration.* Submitted to *Network* (MDPI), 2026.

The repository contains two emulated OpenFlow testbeds, the exhaustive measurement banks collected on them, the fluid-model digital twins, 24 modelled landscapes on Internet Topology Zoo backbones, the iterate–test loop with eleven alternative policies, every experiment (E1–E11) and the scripts that draw every figure of the paper.

## Quick start (no testbed needed)

Every figure and every number of the paper can be reproduced from the stored data:

```bash
pip install -r requirements.txt
./make_figures.sh                      # all 30 figures -> figures/*.pdf
python3 experiments.py landscape       # recomputes results/landscape.json from data/bank.jsonl
```

Run all scripts from the repository root (paths such as `data/bank.jsonl` are relative). Set `FIGDIR=/some/dir` to write figures elsewhere.

## Repository layout

| Path | Content |
|---|---|
| `testbed.py` | Small testbed: 6 Open vSwitch bridges, 8 shaped links, 5 elastic demands, a delay probe, random cross-traffic, 10 binary knobs. Builds the network, compiles a configuration into OpenFlow 1.3 flow and meter tables, installs it with bundles and runs one test. |
| `bigbed.py` | Large testbed: the GÉANT 2012 backbone with 40 bridges, 61 shaped links and 32 host namespaces; only the changed tables are pushed. `data/big_describe.json` lists demands, knobs, sleep groups and cross-traffic. |
| `enumerate_bank.py`, `enumerate_big.py` | Exhaustive measurement banks: 1024 configurations × 5 passes (`data/bank.jsonl`) and × 4 passes (`data/big_bank.jsonl`). |
| `window_noise.py` | Noise against the length of the measurement window (`data/window_noise.jsonl`). |
| `fluid.py`, `twin_testbed.py`, `twin_big.py` | Max–min fluid model and the digital twins of both testbeds (`data/twin_*.npy`). |
| `zooworlds.py` | 24 modelled landscapes (8 Topology Zoo backbones × 3 seeds, 14 knobs, 32 gravity traffic matrices) in `data/zoo/*.npz`. |
| `sdnloop.py` | The iterate–test loop and the alternatives (UCB, pre-screened UCB, Thompson sampling, ε-greedy, hill climbing, random, measure-once, twin only, simulate-then-validate, ablations). |
| `worlds.py` | Loaders that turn banks and landscapes into replay worlds. |
| `experiments.py`, `experiments_big.py` | Replay experiments (`python3 experiments.py <section> ...`; sections: `landscape zoo_stats tune policies_zoo policies_testbed policies_testbed_mid allocation ablation bias dynamics twinq drift`). Results in `results/*.json`. |
| `live.py`, `live_big.py` | Closed-loop runs on the live testbeds (`data/live_*.jsonl`, `data/big_live_*.jsonl`). |
| `split_replay.py`, `split_full.py`, `split_small.py`, `split_small_all.py` | Cross-validated replay: tests drawn from some passes of a bank, the shipped configuration scored against the means of the held-out passes (removes the optimism of scoring a choice against the measurements that guided it). Results in `results/split_*.json`. |
| `rerun_twin_policies.py` | Recomputes the twin-based policies with the twin's ties broken at random (twin only: expectation over tied configurations) and retunes simulate-then-validate. |
| `live_analysis.py`, `shift_replay.py` | Closed-loop runs against the replay: significance of differences between policies, replay with the live noise level and with a configuration-specific shift between bank and live network, transients after a configuration change (`results/live_analysis.json`, `results/shift_replay.json`). |
| `rtt_timing.py`, `rtt_timing_analysis.py` | Probe timing: share of ICMP probes sent after the transfers ended and the effect of averaging the RTT over the whole probe window (`data/rtt_timing.jsonl`, `results/rtt_timing.json`). |
| `restart_servers.py` | Restarts the iperf3 servers of the small testbed after an interrupted run. |
| `tables.py` | Table rows of the paper. |
| `figs.py`, `fig_process.py`, `fig_maps.py`, `fig_topo.py`, `fig_concept.py`, `fig_exp2.py` | Figures. `icons.py` draws the device icons, `schem.py` the schematics of each experiment, with an automatic check that no labels or elements overlap. |
| `zoo/` | GraphML files from the Internet Topology Zoo used for the landscapes and the GÉANT testbed. |
| `data/logs/` | Logs of the measurement campaigns. |

## Running the testbeds (optional)

Needed only to collect new measurements. Requirements: Linux with network namespaces, veth pairs and TBF qdiscs; Open vSwitch 3.x (userspace `netdev` datapath, no kernel module); iperf3, iproute2, ethtool, ping; root privileges. The data in this repository were collected with Open vSwitch 3.3.9, iperf3 3.16 and Python 3.13.

```bash
sudo python3 testbed.py build
sudo python3 testbed.py test 0000010000 1.5      # one test of one configuration
sudo python3 bigbed.py build
sudo python3 bigbed.py test 0000000000
```

A full bank takes about 3 hours of testing on either testbed (5120 tests of about 2.0 s on the small one, 4096 tests of about 2.4 s on GÉANT). iperf3 counts whole seconds, so a requested 1.5 s window runs the TCP and UDP transfers for 1 s and the probe for 1.5 s.

## Data format

Each line of `data/bank.jsonl` and `data/big_bank.jsonl` is one test: the configuration as a 10-character bit string (`x`, character *i* = knob *i*), the utility `u` and its criteria (throughput satisfaction `T`, delay score `L`, energy score `E`), the probe RTT in ms (`rtt`) and loss, the measured rate of every demand in Mbit/s (`thr`), the cross-traffic rates drawn for the test (`bg`), the window length in s (`dur`), the Unix time (`t`) and the pass.

## Licence

Code: MIT (see `LICENSE`). Measurement data and results (`data/`, `results/`): CC BY 4.0. Topology Zoo files (`zoo/`): from the Internet Topology Zoo (Knight et al., IEEE JSAC 29(9), 2011, https://doi.org/10.1109/JSAC.2011.111002); please cite that work when you use them.

## Citation

If you use this code or data, please cite the article above (see `CITATION.cff`). The theory it builds on is Kuraś, P.; Michalski, A.; Gerka, A.; Organiściak, P. *When Does Design Thinking Work? A Convergence Theorem for the Iterate–Test Core*, arXiv:2611.03034, https://arxiv.org/abs/2611.03034.
