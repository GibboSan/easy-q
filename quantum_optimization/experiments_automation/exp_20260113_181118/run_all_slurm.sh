#!/bin/bash
set -e

mkdir -p logs

sbatch --job-name=N_nodes57_N_layers1 --mem=64G --cpus-per-task=8 --time=00:45:00 --mail-type=FAIL --mail-user=gabriele.sanguin.2@studenti.unipd.it --partition=allgroups --account=phd --output=logs/N_nodes57_N_layers1_%j.out --wrap="mkdir -p N_nodes57_N_layers1_out && python3 ../../run_performance.py N_nodes57_N_layers1/N_nodes57_N_layers1.yaml > N_nodes57_N_layers1_out/output.log"
sbatch --job-name=N_nodes60_N_layers1 --mem=64G --cpus-per-task=8 --time=00:45:00 --mail-type=FAIL --mail-user=gabriele.sanguin.2@studenti.unipd.it --partition=allgroups --account=phd --output=logs/N_nodes60_N_layers1_%j.out --wrap="mkdir -p N_nodes60_N_layers1_out && python3 ../../run_performance.py N_nodes60_N_layers1/N_nodes60_N_layers1.yaml > N_nodes60_N_layers1_out/output.log"

