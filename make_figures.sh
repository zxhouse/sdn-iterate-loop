#!/usr/bin/env bash
# Regenerates every figure of the paper from data/ and results/ (no testbed needed).
# Output: figures/*.pdf (override with FIGDIR=...), previews in figures/png/.
set -e
cd "$(dirname "$0")"
python3 fig_concept.py
python3 fig_topo.py
python3 fig_maps.py geant zoo
python3 fig_process.py timeline pipeline livetrace
python3 figs.py twinq landscape dynamics allocation bias big extra curse converge ctrlcost
python3 fig_exp2.py e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 implementation packet_path
